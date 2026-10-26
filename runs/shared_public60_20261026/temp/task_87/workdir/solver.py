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
### question: For patient 031-3355, was the collection time of the blood, venipuncture culture linked to the current stay earlier than the admission time for that visit?
### 可用表 (run_sql 内直接用这些名字): diagnosis, df_diagnosis, treatment, df_treatment, vitalperiodic, df_vitalperiodic, cost, df_cost, allergy, df_allergy, intakeoutput, df_intakeoutput, lab, df_lab, medication, df_medication, microlab, df_microlab, patient, df_patient
### TODO: 根据 question 和 knowledge.md 理解任务, 结合数据源, 完成查询逻辑.
### 在此处用注释写下分析过程: 每个字段的含义/来源/join 逻辑/过滤与聚合粒度/空值处理.
### 分析口径:
### 1) 目标: 判断 patient 031-3355 "当前住院" 关联的 'blood, venipuncture' 血培养采集时间
###    是否早于该次住院的入院时间 -> 输出布尔 Yes/No.
### 2) 表映射:
###    - patient: 提供 uniquepid / patientunitstayid / 入院时间 (hospitaladmittime 入院,
###    (unitadmittime 入科). question 说 "admission time for that visit", 取入院时刻
###    hospitaladmittime (本次就诊开始), 两种口径结论一致, 不敏感.
###    - microlab: 培养记录, 通过 patientunitstayid 关联到具体一次 stay (即 question 中
###      "linked to the current stay"), culturesite = 'blood, venipuncture' 精确取该培养.
###    - doc 抽取的 cost 表: 记录 (cost record -> uniquepid -> patienthealthsystemstayid)
###      的映射与若干"录入纠正"陷阱; 但其 event_type 均为 lab/medication/diagnosis 等,
###      不含 microlab 事件, 且无 031-3355 的记录 (仅一条 record_id=124749 空行),
###      故本任务的 stay 关联只能走 patientunitstayid, 无需 cost 表参与.
### 3) 时间口径: culturetakentime 2105-12-21 14:02:00 同时早于 hospitaladmittime
###    (2105-12-27 11:02) 与 unitadmittime (2105-12-27 18:28), 两种口径结论一致 = Yes.
### 4) 行集: 唯一 1 行布尔结果, 无 NULL, 不需排序/去重.
result = run_sql("""
SELECT
    CASE WHEN m.culturetakentime < p.hospitaladmittime THEN 'Yes' ELSE 'No' END AS is_earlier
FROM df_patient p
JOIN df_microlab m
  ON m.patientunitstayid = p.patientunitstayid
WHERE p.uniquepid = '031-3355'
  AND m.culturesite = 'blood, venipuncture'
""")


## 数据字段转换(如有必要) 根据 question 要求对结果进行转化，输出所需字段


## 保存预测结果
pred_df = result  # TODO: 替换为最终结果 DataFrame
if pred_df is None:
    # 查询尚未实现: 首次执行/未填 SQL 时走这里, 不写 prediction.csv, 也不报错.
    print('[solver] 查询尚未实现 (result is None): 请在 "## 进行查询" 段补全 SQL 后再运行; 本次未写出 prediction.csv')
else:
    pred_df.to_csv(output_base / 'prediction.csv', index=False)  # don't change
