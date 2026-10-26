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
### 口径:
### - 视频 Condition 1/2: Senior cohort = AGE >= 60 (admissions 中 AGE>=60 行数 = 724, 与面板 Cohort size 一致)
### - 视频 Condition 2/2: Report scope = Top 3 (按 procedure 计数降序, 第3名=ICD9 966, 第4名=ICD9 9671 被排除)
### - 目标输出列: procedure_name (D_ICD_PROCEDURES.SHORT_TITLE) + occurrence_count (辅助列, 用于解释排名)
### - 计数粒度: procedures_icd 记录条数, 按 ICD9_CODE 分组; senior cohort 用 SUBJECT_ID IN (AGE>=60 的患者)
### - 排序: n DESC, ICD9_CODE ASC; LIMIT 3
result = run_sql("""
WITH senior AS (
    SELECT DISTINCT SUBJECT_ID FROM admissions WHERE AGE >= 60
),
counts AS (
    SELECT p.ICD9_CODE, COUNT(*) AS n
    FROM procedures_icd p
    JOIN senior s ON p.SUBJECT_ID = s.SUBJECT_ID
    GROUP BY p.ICD9_CODE
)
SELECT d.SHORT_TITLE AS procedure_name, c.n AS occurrence_count
FROM counts c
JOIN d_icd_procedures d ON TRIM(d.ICD9_CODE) = CAST(c.ICD9_CODE AS VARCHAR)
ORDER BY c.n DESC, c.ICD9_CODE ASC
LIMIT 3
""")


## 数据字段转换(如有必要) 根据 question 要求对结果进行转化，输出所需字段


## 保存预测结果
pred_df = result  # TODO: 替换为最终结果 DataFrame
if pred_df is None:
    # 查询尚未实现: 首次执行/未填 SQL 时走这里, 不写 prediction.csv, 也不报错.
    print('[solver] 查询尚未实现 (result is None): 请在 "## 进行查询" 段补全 SQL 后再运行; 本次未写出 prediction.csv')
else:
    pred_df.to_csv(output_base / 'prediction.csv', index=False)  # don't change
