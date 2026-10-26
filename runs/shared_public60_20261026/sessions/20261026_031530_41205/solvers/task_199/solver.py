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
### question: In the fund-manager tenure records, which current fund managers have total management tenure below 0.2 months? List their manager code, name, fund-advisor abbreviation, and total management tenure in months.
### 可用表 (run_sql 内直接用这些名字): 公募基金分红, df_公募基金分红, 公募基金概况, df_公募基金概况, 公募基金管理人概况, df_公募基金管理人概况, 公募基金行业投资, df_公募基金行业投资, 公募基金衍生指标_基金公司收益与规模同类均值与排名, df_公募基金衍生指标_基金公司收益与规模同类均值与排名, 公募基金衍生指标_基金经理最新收益与规模排名, df_公募基金衍生指标_基金经理最新收益与规模排名, 基金公司基金经理人数, df_基金公司基金经理人数, 公募基金托管人概况, df_公募基金托管人概况, 公募基金最新收益率排名, df_公募基金最新收益率排名, 公募基金重仓股票组合, df_公募基金重仓股票组合, 公募基金风险等级表, df_公募基金风险等级表, 基金经理规模统计_新_, df_基金经理规模统计_新_, 公募基金经理_新_, df_公募基金经理_新_, 公募基金经理基本资料, df_公募基金经理基本资料, 公募基金衍生指标_基金经理收益与规模同类分析, df_公募基金衍生指标_基金经理收益与规模同类分析, 公募基金主要财务指标_季报_, df_公募基金主要财务指标_季报_, 公募基金债券组合明细, df_公募基金债券组合明细, 公募基金净值最新区间表现, df_公募基金净值最新区间表现, 公募基金最新基准收益率, df_公募基金最新基准收益率, 公募基金管理人规模及排名情况, df_公募基金管理人规模及排名情况, 公募基金衍生指标_基金公司收益与规模同类分析, df_公募基金衍生指标_基金公司收益与规模同类分析, 公募基金费率_新_, df_公募基金费率_新_, 公募基金资产配置, df_公募基金资产配置, 基金经理任职及管理年限统计, df_基金经理任职及管理年限统计, 基金经理最新任职经验与稳定性指标与排名, df_基金经理最新任职经验与稳定性指标与排名
### TODO: 根据 question 和 knowledge.md 理解任务, 结合数据源, 完成查询逻辑.
### 在此处用注释写下分析过程: 每个字段的含义/来源/join 逻辑/过滤与聚合粒度/空值处理.
### 写法示例: result = run_sql("""SELECT ... FROM df_xxx WHERE ...""")
result = None  # TODO: 替换为上面示例形式的真实查询


## 数据字段转换(如有必要) 根据 question 要求对结果进行转化，输出所需字段


## 保存预测结果
pred_df = result  # TODO: 替换为最终结果 DataFrame
if pred_df is None:
    # 查询尚未实现: 首次执行/未填 SQL 时走这里, 不写 prediction.csv, 也不报错.
    print('[solver] 查询尚未实现 (result is None): 请在 "## 进行查询" 段补全 SQL 后再运行; 本次未写出 prediction.csv')
else:
    pred_df.to_csv(output_base / 'prediction.csv', index=False)  # don't change
