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
### question: How many companies in Jiangsu Province have a free float ratio of 100%?
### 可用表 (run_sql 内直接用这些名字): a股发行申报企业信息, df_a股发行申报企业信息, 公司主营业务构成, df_公司主营业务构成, 公司分红, df_公司分红, 公司实际控制人, df_公司实际控制人, 公司报告期管理层持股, df_公司报告期管理层持股, 公司概况, df_公司概况, 公司股本结构变动, df_公司股本结构变动, 利润分配表附注_财务费用, df_利润分配表附注_财务费用, 限售股票解禁时间表, df_限售股票解禁时间表, 日行情表, df_日行情表, 股东股权冻结和质押, df_股东股权冻结和质押, 股东股权冻结和质押统计, df_股东股权冻结和质押统计, 股东股权变动, df_股东股权变动, 股票交易资金流向, df_股票交易资金流向, 股票月度行情数据, df_股票月度行情数据, 自由流通股本, df_自由流通股本, 配股大股东认配状况, df_配股大股东认配状况, a股增发, df_a股增发, 企业之间参股情况, df_企业之间参股情况, a股询价明细, df_a股询价明细, a股配股, df_a股配股, 公司经营范围与行业变更, df_公司经营范围与行业变更, 公司行业划分表, df_公司行业划分表, 概念所属公司表, df_概念所属公司表, 法人配售与战略投资者, df_法人配售与战略投资者, 股东户数, df_股东户数, 股东持股统计, df_股东持股统计, 股东股权托管, df_股东股权托管, 股份回购, df_股份回购
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
