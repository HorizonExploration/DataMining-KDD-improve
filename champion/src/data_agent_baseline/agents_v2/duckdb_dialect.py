"""DuckDB 方言与查询写法说明 (注入 solver_agent 的 system_instruction).

为什么独立成文件:
- 这段方言提示会注入到每个 solver 的 system prompt, 内容较多且需要持续维护
  (随 DuckDB 版本、踩坑经验增补), 单独成文件比内联在 solver_agent.py 里清晰.

内容依据 DuckDB 官方文档 (https://duckdb.org/docs/current/sql/) 整理, 重点覆盖
LLM 生成 DuckDB SQL 时最易出错的点: 整数除法、正则部分匹配、强类型转换、
中文/特殊字符标识符、空组聚合返回 NULL、极值并列等.
"""

from __future__ import annotations


# DuckDB 方言细节与陷阱 (拼接到 solver_agent system_instruction 末尾).
DUCKDB_DIALECT_NOTE = """- 方言类型：DuckDB SQL。整体接近 PostgreSQL, 强数据类型。
不是 SQLite/MySQL 方言，不支持 MySQL 反引号、SQLite 式宽松类型、SQLite 参数顺序的 strftime。
- explore_data 与 solver.py 均基于 DuckDB 引擎, 探查通过的 SQL 可直接搬进 solver.py.

### DuckDB 标识符与字面量
1. 表名：DataFrame 变量名可直接当表名使用，例如 `df_xxx`、`orders`，无需注册临时表。
2. 标识符：列名/表名只使用双引号，例如 `"列名"`；不支持 MySQL 反引号 `列名` 或 SQL Server 方括号 `[列名]`。
   列名含空格、括号、斜杠、百分号、中文等特殊字符时必须用双引号，并且要与真实列名逐字一致，
   例如 `"应配股数(股)"`、`"三年回报率(%)"`、`"A/B"`。
   特别注意中文列名括号前不要自行加空格(`"在任基金数 (只)"` 是错的, 真实列是 `"在任基金数(只)"`);
   run_sql 会兜底纠正空格差异, 但应优先逐字写对。
   标识符大小写不敏感(即使加双引号), 但会保留原始大小写; 不要靠大小写区分两个列。
3. 字符串字面量必须用单引号，例如 `'2023'`、`'股票'`。双引号表示列名，不表示字符串。
   错写成 `"股票"` 会被 DuckDB 当作列名并报错。

### DuckDB 强类型与转换
4. 文本列与数字/日期比较或计算前要显式转换。
   推荐用 `TRY_CAST(...)` 提高容错，转换失败返回 NULL，不会中断查询；例如：
     `TRY_CAST("金额" AS DOUBLE)`、`TRY_CAST("数量" AS BIGINT)`。
   若列中可能有空字符串，优先写：
     `TRY_CAST(NULLIF(TRIM("金额"), '') AS DOUBLE)`。
5. 除法是浮点除法：`5 / 2 = 2.5`(不是 SQLite/PG 的整除)。需要整除用 `//`：`5 // 2 = 2`。
   结果可能多出 `.0`；若目标是整数列, 用 `//` 或 `CAST(... AS BIGINT)`。
   除法、比例、百分比计算前先把参与列转成 DOUBLE：`TRY_CAST("收益率(%)" AS DOUBLE) / 100.0`。
6. 隐式转换较激进且不直观：`'1.1' = 1` 为 true、`true = 1` 为 true。比较前尽量显式 CAST 到同一类型。
7. CASE / COALESCE / UNION 各分支类型必须一致。
   错误示例：`CASE WHEN cond THEN 1 ELSE '未知' END`。
   应统一字符串：`CASE WHEN cond THEN '1' ELSE '未知' END`，或统一数字：`CASE WHEN cond THEN 1 ELSE NULL END`。

### 日期与时间
8. 字符串日期不能直接当日期函数使用，需先转换。
   ISO 格式日期可用：`CAST("日期" AS DATE)` 或 `TRY_CAST("日期" AS DATE)`。
   提取年份推荐：`YEAR(TRY_CAST("日期" AS DATE))`；普通字符串也可直接取：`SUBSTR("日期", 1, 4)`。
   DuckDB 没有 `to_date`，字符串按自定义格式解析用 `strptime(text, fmt)`(失败报错)
   或 `try_strptime(text, fmt)`(失败返 NULL)，例如 `try_strptime("d", '%d/%m/%Y')`。
   注意 `strftime(date, fmt)` 参数顺序与 SQLite 相反：
     DuckDB: `strftime(date_col, '%Y')`；SQLite: `strftime('%Y', date_col)`。
9. 时间字符串排序：形如 "1:32.456" 的字符串排序前先转成秒(数值)，否则按字典序排错。

### NULL 与聚合
10. NULL 判断必须用 `IS NULL` / `IS NOT NULL`，不要写 `= NULL` 或 `!= NULL`。
    空字符串不是 NULL，如需同时处理：`NULLIF(TRIM(col), '') IS NULL`。
    涉及 normal/abnormal/分类判断时显式 `WHERE col IS NOT NULL`，不要让 NULL 隐式归入某一类。
11. 空组聚合返回 NULL(不是 0/空)：`sum`/`list`/`string_agg` 在无匹配行时返 NULL，只有 `count` 返 0。
    需要把空结果当 0 时用 `COALESCE(SUM(x), 0)`。
12. 聚合查询中 SELECT 里的非聚合字段必须出现在 GROUP BY 中。
    可用 `GROUP BY ALL` 自动按所有非聚合列分组(推荐，避免漏列/重复)；需要任意值用 `ANY_VALUE(col)`。
13. 取每组「val 最大/最小那行的某字段」用 `arg_max(arg, val)` / `arg_min(arg, val)`(别名 `max_by`/`min_by`)，
    比自连接或窗口子查询更简洁；取前 N 用 `arg_max(arg, val, n)` 返回 list。

### 排序、TopN、极值并列
14. 排序取 TopN 用 `ORDER BY ... LIMIT n`；默认 `ASC NULLS LAST`，可显式 `NULLS LAST`/`NULLS FIRST`。
    对某列排序时务必排除 NULL(`WHERE col IS NOT NULL`)。
15. "lowest / highest / fastest / smallest / Nth" 等极值题要考虑**并列**：
    优先 `WHERE col = (SELECT MIN(col) FROM ...)`，而不是 `ORDER BY col LIMIT 1`，
    除非 question 明确说 "the one" / "top 1"。
16. 过滤窗口函数结果用 `QUALIFY`(免子查询，类似 HAVING 之于聚合)：
    `... QUALIFY row_number() OVER (PARTITION BY g ORDER BY t) = 1`。

### 字符串匹配与正则 (RE2 引擎)
17. `LIKE`(`_`=单字符, `%`=任意长度, 整串匹配, 子串匹配要 `%...%`)、`ILIKE`(大小写不敏感)。
18. 关键词模糊匹配(如 "X-related districts/areas/topics")用 `LIKE '%X%'`，
    不要假设有现成字段恰好等于 X。
19. 正则**部分匹配**(含子串即真)用 `regexp_matches(col, pattern)`，
    不要用 `~`(在 DuckDB 中 `~` = `regexp_full_match` 整串全匹配，语义不同)。
    提取用 `regexp_extract(col, pat[, group])`(无匹配返空串)，替换用 `regexp_replace(col, pat, repl[, 'g'])`。
    只支持 9 个捕获组 `\\1`..`\\9`。

### 解题口径
20. 术语严格映射：不要把 question 里的修饰语扩大解释。如 "severe" 在某 schema 里可能恰好
    等于一个具体取值(如 `Thrombosis = 2`)，而不是 `IN (1, 2)`。
21. 多重定语主键定位：如 "year + race name" 未必唯一定位 race，先在子查询里用最严格条件
    定位主键(如 raceId)，单跑一次确认唯一，再写主查询。
22. "average ... per month" 是 `AVG(X)` 还是 `SUM(X)/12`，取决于 X 是月度记录还是已聚合，先看 schema 再决定。
23. 数值题输出纯数值，不要 "$100" / "10 kg" 这种带单位字符串。

### 报错排查
24. 写错列名、类型不匹配或转换失败时 DuckDB 通常直接报错。
    根据报错中的 `Binder Error`(列名/分组问题)、`Catalog Error`(表名/列名不存在)、
    `Conversion Error`(类型转换失败)修正列名、引号或 CAST 即可。"""
