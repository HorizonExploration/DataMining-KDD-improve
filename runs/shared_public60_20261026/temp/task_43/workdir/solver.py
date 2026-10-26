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
### question: 请按本周以来基金基准增长幅度从高到低列出排名前10的基金及其投资目标；若第10名增长幅度并列，保留所有并列基金，并列项按资料中的记录顺序输出。
### 可用表 (run_sql 内直接用这些名字): mf_advisorscalerank, df_mf_advisorscalerank, mf_assetallocation, df_mf_assetallocation, mf_fcretscalerank, df_mf_fcretscalerank, mf_fmperfanalysis, df_mf_fmperfanalysis, mf_fmretscaleanalysis, df_mf_fmretscaleanalysis, mf_fmtermanalysis, df_mf_fmtermanalysis, mf_fundarchives, df_mf_fundarchives, mf_fundrisklevel, df_mf_fundrisklevel, mf_mainfinancialindexq, df_mf_mainfinancialindexq, mf_managerexperience, df_mf_managerexperience, mf_benchmarkgrowthrate, df_mf_benchmarkgrowthrate, mf_fundmanagernew, df_mf_fundmanagernew, mf_investindustry, df_mf_investindustry, mf_keystockportfolio, df_mf_keystockportfolio, mf_netvalue, df_mf_netvalue, mf_awards, df_mf_awards, mf_bondportifoliodetail, df_mf_bondportifoliodetail, mf_dividend, df_mf_dividend, mf_fcnumofmanager, df_mf_fcnumofmanager, mf_fcretscaleanalysis, df_mf_fcretscaleanalysis, mf_fmretandscalerank, df_mf_fmretandscalerank, mf_fundreturnrank, df_mf_fundreturnrank, mf_netvalueperformancehis, df_mf_netvalueperformancehis, mf_personalinfo, df_mf_personalinfo
### TODO: 根据 question 和 knowledge.md 理解任务, 结合数据源, 完成查询逻辑.
### 在此处用注释写下分析过程: 每个字段的含义/来源/join 逻辑/过滤与聚合粒度/空值处理.
### 写法示例: result = run_sql("""SELECT ... FROM df_xxx WHERE ...""")
# 口径: 目标列 = 基金简称 (SecuAbbr) + 投资目标 (InvestTarget), 参考 knowledge.md Use Case 2,
#       Join mf_benchmarkgrowthrate (基准增长率) 与 mf_fundarchives (投资目标 InvestTarget) ON InnerCode.
# "本周以来" -> BenchGRForThisWeek (knowledge 4.4). 排序 DESC, 取前 10.
# 并列处理: 第 10 名若并列则全部保留 -> 用源表记录顺序 row_number 定出第 10 名的值, 再取 g >= 该值.
# NULL: 排名/TopN 必须排除 BenchGRForThisWeek 为 NULL 的记录. InvestTarget 保留原值(含NULL).
result = run_sql("""
WITH r AS (
    SELECT b."SecuAbbr" AS secuabbr,
           b."InvestTarget" AS investtarget,
           a."BenchGRForThisWeek" AS benchgrforthisweek,
           row_number() OVER (ORDER BY (SELECT 1)) AS src_order
    FROM mf_benchmarkgrowthrate AS a
    JOIN mf_fundarchives AS b ON a."InnerCode" = b."InnerCode"
    WHERE a."BenchGRForThisWeek" IS NOT NULL
),
cut AS (
    SELECT benchgrforthisweek AS thr
    FROM (SELECT benchgrforthisweek, row_number() OVER (ORDER BY benchgrforthisweek DESC, src_order) AS rn FROM r)
    WHERE rn = 10
)
SELECT secuabbr, investtarget
FROM r
WHERE benchgrforthisweek >= (SELECT thr FROM cut)
ORDER BY benchgrforthisweek DESC, src_order
""")


## 数据字段转换(如有必要) 根据 question 要求对结果进行转化，输出所需字段


## 保存预测结果
pred_df = result
if pred_df is None:
    # 查询尚未实现: 首次执行/未填 SQL 时走这里, 不写 prediction.csv, 也不报错.
    print('[solver] 查询尚未实现 (result is None): 请在 "## 进行查询" 段补全 SQL 后再运行; 本次未写出 prediction.csv')
else:
    pred_df.to_csv(output_base / 'prediction.csv', index=False)  # don't change
