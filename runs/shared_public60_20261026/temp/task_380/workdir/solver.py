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
### question: Find all non-Chinese fund managers.
### 可用表 (run_sql 内直接用这些名字): mf_advisorscalerank, df_mf_advisorscalerank, mf_fcnumofmanager, df_mf_fcnumofmanager, mf_fcretscaleanalysis, df_mf_fcretscaleanalysis, mf_fmperfanalysis, df_mf_fmperfanalysis, mf_fmretscaleanalysis, df_mf_fmretscaleanalysis, mf_fundmanagernew, df_mf_fundmanagernew, mf_investindustry, df_mf_investindustry, mf_keystockportfolio, df_mf_keystockportfolio, mf_fmtermanalysis, df_mf_fmtermanalysis, mf_fundrisklevel, df_mf_fundrisklevel, mf_investadvisoroutline, df_mf_investadvisoroutline, mf_mainfinancialindexq, df_mf_mainfinancialindexq, mf_trusteeoutline, df_mf_trusteeoutline, mf_benchmarkgrowthrate, df_mf_benchmarkgrowthrate, mf_fmretandscalerank, df_mf_fmretandscalerank, mf_fmscaleanalysisn, df_mf_fmscaleanalysisn, mf_fundreturnrank, df_mf_fundreturnrank, mf_netvalueperformancehis, df_mf_netvalueperformancehis, sub_db__mf_advisorscalerank, df_sub_db__mf_advisorscalerank, mf_assetallocation, df_mf_assetallocation, mf_awards, df_mf_awards, mf_bondportifoliodetail, df_mf_bondportifoliodetail, mf_chargeratenew, df_mf_chargeratenew, mf_dividend, df_mf_dividend, mf_fcretscalerank, df_mf_fcretscalerank, mf_fundarchives, df_mf_fundarchives, mf_managerexperience, df_mf_managerexperience, mf_netvalue, df_mf_netvalue, mf_personalinfo, df_mf_personalinfo
### TODO: 根据 question 和 knowledge.md 理解任务, 结合数据源, 完成查询逻辑.
### 分析过程 (口径决策):
# 题型: 实体名单/成员判定 —— "Find all non-Chinese fund managers" 要的是基金经理实体集合,
#       同一经理在母表 (mf_fundmanagernew 任职明细) 中多行重复, 必须按实体 DISTINCT.
# 目标实体: 基金经理. 主表 mf_fundmanagernew (基金经理任职明细), 取 PostName = '基金经理'
#       (表内另有 11 人 PostName = '基金经理助理', 不属于基金经理; 且其中无外籍, 不影响结果).
# 国籍字段映射: mf_fundmanagernew 无国籍字段, 国籍在 mf_personalinfo.Nationality
#       (经 PersonalCode 关联). 532 名基金经理全部能在 mf_personalinfo 中找到, 无缺失.
# 空值处理 (关键): mf_personalinfo.Nationality 有 3 行为 NULL, 但这 3 人 Background 简介
#       分别写明 "加拿大籍 / 新加坡籍 / 马来西亚籍", 实为外籍, 不能按 NULL 丢弃.
#       全表校验: Nationality='中国' 的 589 人 Background 全部含 '中国籍'; 3 个 NULL + 1 个 '美国'
#       的 Background 全部为境外籍, 两个来源完全一致. 故国籍取
#       COALESCE(NullIf(Nationality,''), regexp_extract(Background, '…籍')) 兜底.
# 过滤: 国籍 <> '中国' (显式排除中国籍, NULL 兜底后不会出现), 不按在任/离任过滤 (题目未限定在任).
# 输出列: 名单题最少必要字段 -> 基金经理姓名 (去重实体名单).

### 在此处用注释写下分析过程: 每个字段的含义/来源/join 逻辑/过滤与聚合粒度/空值处理.
### 写法示例: result = run_sql("""SELECT ... FROM df_xxx WHERE ...""")
result = run_sql("""
    SELECT DISTINCT m.Name AS Name
    FROM mf_fundmanagernew m
    JOIN mf_personalinfo p
      ON p.PersonalCode = m.PersonalCode
    WHERE m.PostName = '基金经理'
      AND COALESCE(
            NULLIF(TRIM(p.Nationality), ''),
            NULLIF(regexp_extract(p.Background, '([^-，,。；;： 0-9A-Za-z()（）]{1,6}籍)', 1), ''),
            ''
          ) <> '中国'
    ORDER BY Name
""")


## 数据字段转换(如有必要) 根据 question 要求对结果进行转化，输出所需字段


## 保存预测结果
pred_df = result
if pred_df is None:
    # 查询尚未实现: 首次执行/未填 SQL 时走这里, 不写 prediction.csv, 也不报错.
    print('[solver] 查询尚未实现 (result is None): 请在 "## 进行查询" 段补全 SQL 后再运行; 本次未写出 prediction.csv')
else:
    pred_df.to_csv(output_base / 'prediction.csv', index=False)  # don't change
