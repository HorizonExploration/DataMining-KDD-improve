"""agents_v2 内部共享工具. 当前只有 ``save_history``.

把 pydantic_ai 的消息历史落到 JSON 文件. 用同样的接口给 solver/pre/post
三个 agent 调用, 避免每个 agent 自己写一份.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai_backends import LocalBackend
from pydantic_ai_backends.permissions import (
    OperationPermissions,
    PermissionRule,
    PermissionRuleset,
)
from pydantic_ai_backends.types import ExecuteResponse

from agents_v2 import tb_fold

_TB_FOLD_OFF = {"0", "false", "no", "off", ""}


def _tb_fold_enabled() -> bool:
    """方案 B 折叠开关, 与 sitecustomize 的方案 A 共用 ``DATA_AGENT_SHORT_TB``.

    默认生效; 仅显式关闭值 (0/false/no/off, 大小写不限) 时不折叠.
    """
    return os.environ.get("DATA_AGENT_SHORT_TB", "1").strip().lower() not in _TB_FOLD_OFF


# 匹配"会切换工作目录的 cd": 命令开头, 或 &&/;/| 之后紧跟的 `cd <非空参数>`.
# 只拦真正的 cd 命令, 不误伤含 "cd" 子串的词(abcd/cdr)或引号/heredoc 内容里的 cd.
# `cd` 单独出现(回 HOME)同样拦, 因为也会改变工作目录.
_CD_RE = re.compile(r"(?:^|[\n;&|]|&&)\s*cd(?:\s+[^\n;&|]*)?(?=$|[\s;&|])", re.MULTILINE)

_CD_BLOCK_MSG = (
    "[blocked] 检测到 `cd` 命令: execute 工具禁止切换工作目录, 始终在任务根目录 `./` 下执行.\n"
    "请去掉 `cd ...`, 改用相对 `./` 的路径:\n"
    "  - 运行求解脚本: `python workdir/solver.py`\n"
    "  - 临时探查: 直接 `python3 << 'PY' ... PY`(脚本内用 './context/...' 等相对路径读数据)\n"
    "  - 看文件/目录: 用 read_file / ls / glob 工具, 路径相对 `./`.\n"
    "注意: 结构化数据分析应优先用 explore_data 工具或 solver.py 的 run_sql, 不要在 execute 里反复手写查询脚本."
)


# 匹配 raw sqlite 访问: python 的 `import sqlite3` / `sqlite3.connect(...)`, 以及命令行
# `sqlite3 <db>`. agent 在 execute 里直连 sqlite 时, 必须手写表名, 而中文/含括号的表名
# (如 `公募基金经理(新)`)极易被臆造出多余空格 -> "no such table" 死循环耗尽配额.
# 结构化数据探查统一走 explore_data(duckdb, 注册了干净别名), 故在工具层硬拦 sqlite3.
# 注意: 用单词边界, 不误伤 `sqlite3-foo` 之类子串; `sqlite_scan`/`pysqlite` 不在拦截内.
_SQLITE_RE = re.compile(r"\bsqlite3\b")

# 匹配直读 context 原始数据源: pandas 的 read_csv/read_json/read_parquet/read_excel/
# read_table/read_sql 且路径指向 context(原始 csv/json/db). 这类直读绕开了 explore_data
# 的列名/表名归一化与 solver.py 的 run_sql 保护, agent 易在手写路径/表名/列名时出错.
# 只拦指向 context 的读取; 读 workdir 下自己产出的中间结果(如 prediction.csv)不拦.
_RAW_READ_RE = re.compile(
    r"""pd\s*\.\s*read_(?:csv|json|parquet|excel|table|sql)\s*\(   # pandas 读取调用
        [^)]*?                                                     # 参数(非贪婪)
        (?:context|input_base)                                     # 指向 context 原始源
    """,
    re.VERBOSE,
)

_SQLITE_BLOCK_MSG = (
    "[blocked] 检测到 raw sqlite 访问(sqlite3): execute 工具禁止直连 sqlite 数据库.\n"
    "中文/含括号空格的表名(如 `公募基金经理(新)`)在手写 SQL 时极易写错表名导致 'no such table'.\n"
    "结构化数据(csv/json/sqlite)的探查请统一用 `explore_data` 工具(后端 DuckDB, 已注册干净表名/别名),\n"
    "最终分析逻辑写回 `./workdir/solver.py` 用其中的 `run_sql(...)` 执行(sqlite 已由 scaffold 经 DuckDB ATTACH 读入为 df_xxx)."
)

_RAW_READ_BLOCK_MSG = (
    "[blocked] 检测到在 execute 里直读 context 下的原始数据文件.\n"
    "原始 csv/json/db 的探查请统一用 `explore_data` 工具(后端 DuckDB, 自动处理表名/列名空格), 而不是在 execute 里手写 pandas 读取.\n"
    "solver.py 已通过 scaffold 把所有数据源读为 df_xxx 变量, 最终分析用 solver.py 的 `run_sql(...)`."
)


def _wrap_backend_block_cd(backend: LocalBackend) -> LocalBackend:
    """包一层 execute: (1) 拦截切目录的 `cd`; (2) 折叠真实输出里的长 traceback (方案 B).

    agent 常无视 prompt 里"不要 cd"的约束(如反复 `cd / && python3 ...`), 在工具层
    硬拦比 prompt 软约束可靠. 拦截走 ExecuteResponse(exit_code=1), console 层会原样
    把提示作为 "Command failed" 回给 agent, 引导它改用 `./` 相对路径 / explore_data.

    真正执行的命令, 其 stdout+stderr 里若含长 traceback(pandas/sqlalchemy 等深层库
    动辄 20+ 行), 在返回前用 :mod:`agents_v2.tb_fold` 折叠中间帧, 只留用户帧+报错点+
    异常消息. 这是方案 A(sitecustomize 拦未捕获异常) 的兜底: agent 自己 try/except 后
    打印的栈、A 漏掉的场景, 都在 execute 这个必经汇聚点被折. 两者共用同一阈值, 幂等.
    """
    _orig_execute = backend.execute

    def _guarded_execute(command: str, timeout=None) -> ExecuteResponse:
        if _CD_RE.search(command):
            return ExecuteResponse(output=_CD_BLOCK_MSG, exit_code=1, truncated=False)
        if _SQLITE_RE.search(command):
            return ExecuteResponse(output=_SQLITE_BLOCK_MSG, exit_code=1, truncated=False)
        if _RAW_READ_RE.search(command):
            return ExecuteResponse(output=_RAW_READ_BLOCK_MSG, exit_code=1, truncated=False)
        resp = _orig_execute(command, timeout)
        if _tb_fold_enabled() and isinstance(resp.output, str) and resp.output:
            try:
                resp.output = tb_fold.fold_text(resp.output)
            except Exception:
                # 折叠失败时保留原始输出, 绝不因美化而丢报错信息.
                pass
        return resp

    backend.execute = _guarded_execute  # type: ignore[method-assign]
    return backend



def build_workdir_local_backend(
    task_dir: Path,
    workdir_name: str = "workdir",
    enable_execute: bool = False,
) -> LocalBackend:
    """构造一个 LocalBackend, 仅允许写入 ``task_dir/<workdir_name>``.

    其他目录(context/, task.json 等)只读. 读/glob/grep/ls 全部放开,
    写和编辑只允许命中 workdir 下的 fnmatch 模式.

    通用 shell ``execute`` 已退化: solver agent 不再挂 execute 工具, 改用专用的
    ``run_solver`` 工具(``tools_v2.run_solver_tool``)跑 ``workdir/solver.py``. 故默认
    ``enable_execute=False`` 关闭 backend 的 shell 执行; grep/glob/ls 由 LocalBackend
    各自独立的 subprocess 实现, 不受此开关影响.

    参数:
        task_dir: LocalBackend 的 root_dir, 一般是当前 task 的临时目录.
        workdir_name: 允许写入的子目录名, 默认 ``workdir``.
        enable_execute: 是否允许 backend 执行命令, 默认 False(已退化).
    """
    task_dir = Path(task_dir)
    workdir_abs = (task_dir / workdir_name).resolve()
    # fnmatch 风格 glob, ** 表示递归; LocalBackend 会先把路径解析到 root_dir 再匹配.
    workdir_patterns = [
        f"{workdir_abs}",
        f"{workdir_abs}/**",
    ]
    write_rules = [
        PermissionRule(pattern=p, action="allow", description="Writes confined to workdir")
        for p in workdir_patterns
    ]
    permissions = PermissionRuleset(
        default="deny",
        read=OperationPermissions(default="allow"),
        write=OperationPermissions(default="deny", rules=write_rules),
        edit=OperationPermissions(default="deny", rules=write_rules),
        execute=OperationPermissions(default="allow"),
        glob=OperationPermissions(default="allow"),
        grep=OperationPermissions(default="allow"),
        ls=OperationPermissions(default="allow"),
    )
    backend = LocalBackend(
        root_dir=task_dir,
        enable_execute=enable_execute,
        permissions=permissions,
    )
    # execute 通道已退化, 不再需要 _wrap_backend_block_cd 的 cd/sqlite3/read_csv
    # 拦截与 traceback 折叠(后者已移到 run_solver_tool). 直接返回裸 backend.
    # 若调用方显式 enable_execute=True 仍想要旧的拦截+折叠行为, 自行包一层.
    return backend


def save_history(history: list, path: Path) -> None:
    """将 pydantic_ai 对话历史序列化为可读 JSON 并保存到 ``path``.

    自动创建父目录. 任何 IO/序列化错误均向上抛, 调用方自行决定如何处理.
    """
    data = ModelMessagesTypeAdapter.dump_python(history, mode="json")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
