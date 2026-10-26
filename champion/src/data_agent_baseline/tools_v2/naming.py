"""数据源文件名 -> 标识符 的统一命名工具.

scaffold_tool(生成 solver.py 的 DataFrame 变量名)与 explore_tool(向 DuckDB
注册表名/别名)必须用**完全相同**的命名规则, 否则 agent 在 explore_data 里
验证过的表名搬进 solver.py 会指向不存在或不同的表 (table_not_found / 数据错配).

历史上两个模块各维护一份 ``_stem_to_varname`` / ``_stem_to_ident``, 靠注释提醒
"必须一致", 容易漂移. 现在统一收口到这里, 两边都调用 ``stem_to_ident``.
"""

from __future__ import annotations

import re


def stem_to_ident(stem: str) -> str:
    """把文件名 stem 转成合法且唯一可辨的 SQL/Python 标识符 (lower-cased).

    用 Unicode 语义的 ``\\w`` 清洗: 保留中文/字母/数字/下划线, 只把标点、空格、
    斜杠、括号等非法字符替换成 ``_``. 中文表名因此得到可辨识且唯一的标识符
    (如 ``配股大股东认配状况``), 避免旧逻辑 ``[^A-Za-z0-9_]`` 把所有中文塌缩成
    纯下划线串导致的两个问题:

    1. 变量名不可辨识, 且 agent 无法从中文表名反推出对应变量名 (只能猜下划线个数);
    2. 不同中文表塌缩成相同串 -> 重名 -> 重名表的 ``df_`` 别名被判冲突而丢弃.

    ``df_<中文>`` 是合法 Python 标识符, DuckDB / pandasql 也都支持中文标识符
    (已验证), 故保留中文安全.
    """
    name = re.sub(r"[^\w]", "_", stem, flags=re.UNICODE)
    if name and name[0].isdigit():
        name = "t_" + name
    return name.lower()


# ---------------------------------------------------------------------------
# SQL 引号方言归一化: 反引号 / 方括号 -> 双引号
# ---------------------------------------------------------------------------
#
# 背景: qwen3.5 带 SQLite/MySQL 方言惯性, 常把标识符写成 MySQL 反引号 ``\`列名\``
# 或 SQL Server / SQLite 方括号 ``[列名]``. DuckDB 只认双引号标识符, 反引号直接
# 触发 ParserError(``syntax error at or near "%"`` 等), 方括号当标识符也报错.
# 实测(6 个 run 的 trajectory)反引号 ~20 条真语法错, 集中在 task_23/30 等.
# prompt 已明令禁止仍照犯(probe 实证: prompt 属弱解), 故在执行边界硬改写.
#
# 安全性:
# - 反引号: DuckDB 里反引号**本就非法**, 全量 ``\`x\`` -> ``"x"`` 不可能破坏原本能
#   跑的 SQL. 唯一要避开的是单引号字符串字面量内部的反引号(不能动).
# - 方括号: DuckDB 用 ``[]`` 做 list 索引/切片(``arr[1]``、``l[1:2]``), 不能误改.
#   故仅在内容"像标识符"(含字母/中文、非纯数字、无冒号)且**不是紧跟在值后面的
#   索引语境**时才改写. 单引号字符串内部一律不动.

# 单引号字符串字面量(含 '' 转义); 用于在扫描时跳过字符串内容.
_SQUOTE_RE = re.compile(r"'(?:[^']|'')*'")
# 反引号标识符(不跨行).
_BACKTICK_RE = re.compile(r"`([^`\n]*)`")
# 方括号: 捕获前一个字符(判断是否索引语境)与括号内容.
_BRACKET_RE = re.compile(r"(.?)\[([^\]\n]*)\]")
# 方括号内容"像标识符"的判定: 含至少一个字母/中文, 不含冒号(切片), 非纯数字.
_IDENT_LIKE_RE = re.compile(r"[A-Za-z一-鿿]")


def _quote_ident(inner: str) -> str:
    """把标识符文本包成 DuckDB 双引号标识符(内部 " 转义为 "")."""
    return '"' + inner.replace('"', '""') + '"'


def _bracket_is_index_context(prev_char: str) -> bool:
    """方括号**紧贴**在值/标识符结尾之后, 则 ``[...]`` 是 list 索引, 不可改写.

    DuckDB list 索引形如 ``arr[1]`` / ``func(...)[1]`` / ``"col"[1]`` / ``l[...][...]``,
    其方括号**无空格地**紧跟在标识符字符、``)``、``]`` 或 ``"`` 之后. SQL Server /
    SQLite 的 ``[标识符]`` 引用则前面要么是空白、要么是 ``(`` / ``,`` / 运算符 /
    行首. 故索引语境只看"紧邻前一字符"(调用方传入 i-1 处字符, 不跳空格):
    一旦中间有空格(``SELECT [col]``)就不是索引, 应改写.
    """
    return bool(prev_char) and (prev_char.isalnum() or prev_char in '_)]"一')


def normalize_quote_dialect(sql: str) -> tuple[str, list[tuple[str, str]]]:
    """把 SQL 里 MySQL 反引号 / SQL Server 方括号标识符改写成 DuckDB 双引号.

    单引号字符串字面量内部一律不动. 方括号仅在"像标识符且非索引语境"时改写,
    避免误伤 DuckDB 的 list 索引/切片 ``arr[1]`` / ``l[1:2]``.

    Returns:
        (改写后的 SQL, [(原写法, 改写为), ...]). 无改写时返回原串和空列表.
    """
    changes: list[tuple[str, str]] = []
    out: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        # 单引号字符串: 原样跳过整段(含 '' 转义), 内部不改写.
        if ch == "'":
            m = _SQUOTE_RE.match(sql, i)
            if m:
                out.append(m.group(0))
                i = m.end()
                continue
            out.append(ch)
            i += 1
            continue
        # 已有的双引号标识符: 原样跳过(交给后续列名归一化处理).
        if ch == '"':
            j = i + 1
            while j < n:
                if sql[j] == '"':
                    if j + 1 < n and sql[j + 1] == '"':
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            out.append(sql[i:j])
            i = j
            continue
        # 反引号标识符 -> 双引号(零风险).
        if ch == "`":
            m = _BACKTICK_RE.match(sql, i)
            if m:
                inner = m.group(1)
                repl = _quote_ident(inner)
                changes.append(("`" + inner + "`", repl))
                out.append(repl)
                i = m.end()
                continue
            out.append(ch)
            i += 1
            continue
        # 方括号: 仅在像标识符且非索引语境时改写.
        if ch == "[":
            end = sql.find("]", i)
            if end != -1 and "\n" not in sql[i:end]:
                inner = sql[i + 1 : end]
                # 紧邻前一字符(不跳空格): 决定是否 list 索引语境.
                prev = sql[i - 1] if i > 0 else ""
                ident_like = (
                    inner != ""
                    and ":" not in inner
                    and _IDENT_LIKE_RE.search(inner) is not None
                )
                if ident_like and not _bracket_is_index_context(prev):
                    repl = _quote_ident(inner)
                    changes.append(("[" + inner + "]", repl))
                    out.append(repl)
                    i = end + 1
                    continue
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), changes


# ---------------------------------------------------------------------------
# SQL 列名空格归一化改写
# ---------------------------------------------------------------------------
#
# 背景: LLM 写中文列名时, 习惯在括号前加空格(如真实列 `在任基金数(只)` 写成
# `在任基金数 (只)`), 或对真实含空格的列(如 `社会消费品零售总额 (百万元)`)漏写
# 空格. 这类纯空格差异在 DuckDB(强类型, 双引号=标识符)下直接报 "column not
# found", agent 反复试错耗尽 request 配额. prompt 提示属弱解(模型照犯), 故在
# SQL **执行边界**做容错: 当双引号标识符原样匹配不上、但"去空格"后能唯一匹配某
# 真实列时, 自动改写成真实列名.
#
# 安全性: 只改写双引号 "..." 内的内容. 在 DuckDB 里双引号永远是标识符(单引号才
# 是字符串字面量), 故改写不会误伤字符串. 只在"去空格后唯一匹配"时改写, 原样能
# 匹配 / 多个候选 / 无候选 时一律不动, 不影响正常查询.

# 匹配双引号标识符: "..."(允许内部转义的 "")
_DQUOTED_RE = re.compile(r'"((?:[^"]|"")*)"')


def _norm_col(name: str) -> str:
    """列名归一化: 去掉所有空白字符(半角/全角空格). 大小写/标点保留(列名大小写敏感)."""
    return re.sub(r"\s+", "", name).replace("　", "")


def rewrite_quoted_columns(sql: str, real_columns) -> tuple[str, list[tuple[str, str]]]:
    """把 SQL 中"去空格后能唯一匹配真实列"的双引号标识符改写成真实列名.

    Args:
        sql: 原始 SQL.
        real_columns: 所有真实列名的可迭代集合(跨表合并即可).

    Returns:
        (改写后的 SQL, [(原写法, 改写为), ...] 改写记录列表).
        无改写时返回原 SQL 和空列表.

    规则:
      - 双引号内容原样就是某真实列 -> 不动(避免无谓改写).
      - 否则按归一化(去空格)匹配真实列: 唯一命中才改写; 0 或多个命中不动.
    """
    real_set = set(real_columns)
    # 归一化 -> 真实列名; 同一归一化键有多个真实列时标记为歧义(值设为 None)
    norm_map: dict[str, str | None] = {}
    for c in real_set:
        k = _norm_col(c)
        if k in norm_map and norm_map[k] != c:
            norm_map[k] = None  # 歧义, 禁止改写
        else:
            norm_map.setdefault(k, c)

    changes: list[tuple[str, str]] = []

    def _sub(m: re.Match) -> str:
        inner = m.group(1).replace('""', '"')  # 还原转义
        if inner in real_set:
            return m.group(0)  # 原样匹配, 不动
        target = norm_map.get(_norm_col(inner))
        if target and target != inner:
            changes.append((inner, target))
            return '"' + target.replace('"', '""') + '"'
        return m.group(0)

    return _DQUOTED_RE.sub(_sub, sql), changes

