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
### question: 请按基金代码升序列出各基金的基金代码、基金简称及今年以来回报率；同一基金出现不同代码时，以对外公示代码为准。请按“基金代码、基金简称、今年以来回报率(%)”三列返回。回报率按四舍五入保留至多 10 位小数，删除末尾多余的 0；整数保留 1 位小数。
### 可用表 (run_sql 内直接用这些名字): mf_advisorscalerank, df_mf_advisorscalerank, mf_fcnumofmanager, df_mf_fcnumofmanager, mf_fcretscaleanalysis, df_mf_fcretscaleanalysis, mf_fmperfanalysis, df_mf_fmperfanalysis, mf_fmretscaleanalysis, df_mf_fmretscaleanalysis, mf_fundmanagernew, df_mf_fundmanagernew, mf_investindustry, df_mf_investindustry, mf_keystockportfolio, df_mf_keystockportfolio, mf_netvalueperformancehis, df_mf_netvalueperformancehis, mf_fmtermanalysis, df_mf_fmtermanalysis, mf_fundrisklevel, df_mf_fundrisklevel, mf_investadvisoroutline, df_mf_investadvisoroutline, mf_mainfinancialindexq, df_mf_mainfinancialindexq, mf_trusteeoutline, df_mf_trusteeoutline, mf_benchmarkgrowthrate, df_mf_benchmarkgrowthrate, mf_fundreturnrank, df_mf_fundreturnrank, mf_assetallocation, df_mf_assetallocation, mf_awards, df_mf_awards, mf_bondportifoliodetail, df_mf_bondportifoliodetail, mf_chargeratenew, df_mf_chargeratenew, mf_dividend, df_mf_dividend, mf_fcretscalerank, df_mf_fcretscalerank, mf_fundarchives, df_mf_fundarchives, mf_managerexperience, df_mf_managerexperience, mf_netvalue, df_mf_netvalue, mf_personalinfo, df_mf_personalinfo
### TODO: 根据 question 和 knowledge.md 理解任务, 结合数据源, 完成查询逻辑.
### 在此处用注释写下分析过程: 每个字段的含义/来源/join 逻辑/过滤与聚合粒度/空值处理.
### 口径分析:
### 1) 题型: 列出类 —— 列出各基金的 (基金代码, 基金简称, 今年以来回报率), 不去重/不过滤.
### 2) 数据源: df_mf_netvalueperformancehis (12962 行, 每行 = 一只基金份额, InnerCode/SecuCode 均唯一).
###    - 今年以来回报率 -> knowledge.md 6.4: "今年以来" 严格对应 RRSinceThisYear (非 RRInSingleYear / RRSinceStart).
###    - 基金简称 -> SecuAbbr (对外公示的证券简称, 与档案表 mf_fundarchives.SecuAbbr 一致).
### 3) "同一基金出现不同代码时, 以对外公示代码为准":
###    跨 12 张表按 SecuAbbr 聚合 SecuCode 做全量排查, 仅 3 只基金存在双代码:
###      华夏创新未来 501207J / 501207, 汇添富创新未来 501206J / 501206, 中欧创新未来 501208J / 501208.
###    带 J 后缀为内部代码, 对外公示(交易所)代码为 6 位纯数字, 即 mf_fundarchives.SecuCode.
###    故对 6位数字+J 的 SecuCode 剥掉 J 后缀; 替换后代码在本表内仍唯一(已验证无重复), 不引入重复行.
### 4) 空值: RRSinceThisYear 无 NULL, 无需过滤.
### 5) 排序: 全部代码为等长 6 位, 按代码字符串升序即数值升序.
### 6) 数值格式: 四舍五入到至多 10 位小数, 去掉末尾多余的 0; 整数值保留 1 位小数 (如 -20.0 / 0.0).
###    用 printf('%.10f', ROUND(r,10)) 得到确定性的 10 位小数字符串(非 float 最短表示),
###    再 rtrim 去尾随 0 与小数点; 若结果不含小数点(即整数)则补 '.0';
###    并对 -0.0xxx 四舍五入成 0 的情形归一为 '0.0', 避免出现 '-0.0'.
result = run_sql("""
WITH base AS (
  SELECT CASE WHEN "SecuCode" SIMILAR TO '[0-9]{6}J' THEN left("SecuCode", 6)
              ELSE "SecuCode" END AS code,
         "SecuAbbr"        AS abbr,
         "RRSinceThisYear" AS r
  FROM df_mf_netvalueperformancehis
)
, fmt AS (
  SELECT code, abbr,
         rtrim(rtrim(printf('%.10f', ROUND(r, 10)), '0'), '.') AS t1
  FROM base
)
SELECT code AS "基金代码",
       abbr AS "基金简称",
       CASE WHEN t1 NOT LIKE '%.%' AND ltrim(t1, '-') = '0' THEN '0.0'
            WHEN t1 NOT LIKE '%.%'                            THEN t1 || '.0'
            ELSE t1
       END AS "今年以来回报率(%)"
FROM fmt
ORDER BY code
""")


## 数据字段转换(如有必要) 根据 question 要求对结果进行转化，输出所需字段
# 基金代码/回报率均为【格式化字符串】: 代码需保留 6 位前导零(000001), 回报率需按
# "至多10位小数/去尾随0/整数留1位小数" 的字面形式落盘. pandas 推断 dtype 会把它们
# 变成 int64 / float64, 读回时前导零与格式全丢, 故全部强制转为 object(str).
if result is not None:
    result = result.astype({'基金代码': 'object', '基金简称': 'object',
                            '今年以来回报率(%)': 'object'})

## 保存预测结果
pred_df = result  # TODO: 替换为最终结果 DataFrame
if pred_df is None:
    # 查询尚未实现: 首次执行/未填 SQL 时走这里, 不写 prediction.csv, 也不报错.
    print('[solver] 查询尚未实现 (result is None): 请在 "## 进行查询" 段补全 SQL 后再运行; 本次未写出 prediction.csv')
else:
    pred_df.to_csv(output_base / 'prediction.csv', index=False)  # don't change
    # 落盘后按 dtype=str 回读自检: 代码唯一/6位数字/升序/无J残留/回报率格式/行数一致
    _chk = pd.read_csv(output_base / 'prediction.csv', dtype=str)
    import re as _re
    _pat = _re.compile(r'^-?(?:0|[1-9]\d*)\.\d{1,10}$')
    _tail = int(_chk['今年以来回报率(%)'].str.contains(r'0$', regex=True).sum()
                - _chk['今年以来回报率(%)'].str.contains(r'\.0$', regex=True).sum())
    print('[solver] shape', _chk.shape,
          '| dup', int(_chk['基金代码'].duplicated().sum()),
          '| non6digit', int((~_chk['基金代码'].str.fullmatch(r'\d{6}')).sum()),
          '| asc', bool(_chk['基金代码'].is_monotonic_increasing),
          '| badfmt', int((~_chk['今年以来回报率(%)'].str.match(_pat)).sum()),
          '| trailing0', _tail,
          '| src_rows', run_sql('SELECT COUNT(*) c FROM df_mf_netvalueperformancehis')['c'][0])
