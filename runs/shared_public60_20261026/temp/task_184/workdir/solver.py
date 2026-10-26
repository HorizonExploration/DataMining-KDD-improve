# this file path is `./workdir/solver.py`
from pathlib import Path
import pandas as pd

# 数据源统一读取 + SQL 方言归一由共享运行时提供, 与 explore_data 工具同一引擎
# 同一列名归一逻辑 —— 在 explore_data 里验证通过的 SQL, 写进 run_sql 即可执行通过.
from tools_v2.datasource_runtime import build_datasource_state, run_sql as _run_sql


## 输入输出路径, 不要修改
input_base = Path('./context')   # use this, don't change
output_base = Path('./workdir')  # use this, don't change


## 读取数据源 (csv/json/sqlite 全部注册为 DuckDB 视图, 表名见下方 schema 速览)
## 不要修改这两行, 也不要再手写 pd.read_csv / sqlite3 直读原始文件.
_state = build_datasource_state(Path('.').resolve())  # task_dir = 当前工作目录


def run_sql(query):
    """执行 SQL 返回完整 DataFrame; 自动处理引号方言(反引号/方括号->双引号)
    与列名多余/缺失空格(归一到真实列名). 直接写真实表名/列名即可.
    表名用 scaffold 注册的名字 (canonical 或 df_<名>), 见下方 schema 速览."""
    return _run_sql(_state, query)


## 数据 schema 速览（仅首次执行打印: 表名 + 列名:类型）
## 需要某表的样本/空值分布时, 用 explore_data 工具的 `\schema <表名>` 按需查
## 不需要修改本段, 也不要删除 workdir/_schema_shown.flag
_COL_MAX = 80  # 单列 '列名:类型' 最大字符; 仅截断过长的个别列, 不影响同行其他列
_schema_flag = Path(__file__).resolve().parent / '_schema_shown.flag'  # 与 solver.py 同目录
if not _schema_flag.exists():
    for _t in _state.tables:
        try:
            _info = _state.conn.execute(f'DESCRIBE "{_t.canonical}"').fetchall()
        except Exception as _e:
            print(f'{_t.canonical}: <describe failed: {_e}>'); continue
        _parts = []
        for _row in _info:
            _c, _ty = str(_row[0]), str(_row[1])
            _entry = f'{_c}:{_ty}'
            if len(_entry) > _COL_MAX:
                _entry = _entry[:_COL_MAX] + f'…(此列名共{len(_entry)}字符,已截断{len(_entry)-_COL_MAX})'
            _parts.append(_entry)
        _alias = f" [别名 df_{_t.canonical}]" if f'df_{_t.canonical}' in _t.aliases else ''
        print(f'{_t.canonical} ({_t.source_type}, {len(_parts)}cols){_alias}: ' + ', '.join(_parts))
    _schema_flag.write_text('1')  # 标记已展示, 后续执行不再重复打印


## 进行查询
### question: What are the funds and fund investment scopes with benchmark growth rate greater than 0 on the current day?
### 口径决策 (依据视频 slide #5/#8: 字段=本日基金基准增长率(%), 运算符 = ">" 严格大于(不含等号), 阈值 = 0;
###   视频 slide #9/#10: 两张表通过 基金内部编码 JOIN, 输出字段 = 基金简称 + 基金投资范围):
### - 目标输出列: fund_abbr (基金简称) 与 fund_investment_scope (基金投资范围), 均为题面明确要求的实体+属性列.
### - 数据源: 公募基金最新基准收益率 (基准增长率表, 1048 行, 基金内部编码唯一) JOIN 公募基金概况 (概况表, 基金内部编码唯一)
### - 过滤: "本日基金基准增长率(%)" > 0 => 严格大于 0, 零增长(0.0)与负增长排除; NULL 自然被比较排除 (该列仅 1 个 NULL).
### - 粒度: 基金内部编码在两张表中均唯一, 结果 374 行且编码唯一, 无需去重.
### - 输出列名沿用视频输出字段: 基金简称 (取概况表 证券简称, 与基准表的 基金简称 一致) 与 基金投资范围.
result = run_sql("""
SELECT
    g."证券简称" AS fund_abbr,
    g."基金投资范围" AS fund_investment_scope
FROM "公募基金最新基准收益率" b
JOIN "公募基金概况" g
  ON b."基金内部编码" = g."基金内部编码"
WHERE b."本日基金基准增长率(%)" > 0
""")


## 数据字段转换(如有必要) 根据 question 要求对结果进行转化，输出所需字段


## 保存预测结果
pred_df = result  # TODO: 替换为最终结果 DataFrame
if pred_df is None:
    # 查询尚未实现: 首次执行/未填 SQL 时走这里, 不写 prediction.csv, 也不报错.
    print('[solver] 查询尚未实现 (result is None): 请在 "## 进行查询" 段补全 SQL 后再运行; 本次未写出 prediction.csv')
else:
    pred_df.to_csv(output_base / 'prediction.csv', index=False)  # don't change
