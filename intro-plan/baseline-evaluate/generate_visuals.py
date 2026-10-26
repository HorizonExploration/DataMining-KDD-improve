"""Generate the baseline report's figures, statistics and offline dashboard.

Run from any directory: python intro-plan/baseline-evaluate/generate_visuals.py
Requires matplotlib; reads only committed result records, never gold or an API.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / 'results/public60_20261026'
FIGURES = HERE / 'figures'
FIGURES.mkdir(exist_ok=True)
font = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11, 'figure.facecolor': '#f6f8fc',
                     'axes.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 180, 'svg.fonttype': 'path'})
BLUE, GREEN, ORANGE, RED = '#3478bc', '#26846c', '#d28a30', '#b74747'
summary = json.loads((RESULTS / 'summary.json').read_text())
evaluation = json.loads((RESULTS / 'evaluation.json').read_text())
course = {r['task_id']: r for r in evaluation['course_results']}
official = {r['task_id']: r for r in evaluation['official_results']}
catalog = {r['task_id']: r['question'] for r in json.loads((HERE / 'task_catalog.json').read_text())}
clinical = {f'task_{n}' for n in (127, 133, 329, 409, 44, 48, 87)}
macro = {f'task_{n}' for n in (10, 168, 208, 264, 312, 328, 336, 35, 389, 391, 58, 86)}
with (RESULTS / 'task_results.csv').open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
for row in rows:
    task = row['task_id']
    row.update(question=catalog[task], domain='医疗' if task in clinical else '宏观经济' if task in macro else '证券与基金',
               modality='含视频' if row['has_video'] == 'True' else '无视频',
               score=float(row['course_score_100']), passed=row['official_passed'] == 'True',
               failure=official[task].get('error_code') or '',
               matched_cols=course[task]['matched_cols'],
               latency_minutes=float(row['last_available_attempt_wall_seconds'] or 0) / 60)
rows.sort(key=lambda r: int(r['task_id'].split('_')[1]))


def group_stats(group):
    return {'tasks': len(group), 'mean_course_score': sum(r['score'] for r in group) / len(group),
            'official_passed': sum(r['passed'] for r in group),
            'course_full': sum(r['score'] == 100 for r in group),
            'course_partial': sum(0 < r['score'] < 100 for r in group),
            'course_zero': sum(r['score'] == 0 for r in group)}


groups = {name: group_stats([r for r in rows if r['domain'] == name]) for name in ('证券与基金', '宏观经济', '医疗')}
partial_points = sum(r['score'] for r in rows if 0 < r['score'] < 100) / 60
assert len(rows) == 60 and abs(sum(r['score'] for r in rows) / 60 - summary['final_course_score_100']) < 1e-8
stats = {'domain_definition': 'Manual question-based grouping; not an official taxonomy', 'domains': groups,
         'reference_score': summary['reference_baseline_100'], 'current_score': summary['final_course_score_100'],
         'reference_model_counts_from_ppt': {'ecnu-max': 32, 'ecnu-plus': 28},
         'current_model_counts': {'retained_ecnu-plus': 33, 'ecnu-max': 27},
         'reference_gap_points': summary['gain_over_reference_points'],
         'full_task_contribution_points': 60.0, 'partial_task_contribution_points': partial_points,
         'zero_task_loss_points': 15.0, 'partial_task_loss_points': 100 - 60 - partial_points - 15,
         'domain_task_ids': {name: [r['task_id'] for r in rows if r['domain'] == name] for name in groups}}
stats['answer_shapes'] = {
    name: group_stats([r for r in rows if predicate(r)])
    for name, predicate in [('single_row', lambda r: int(r['gold_rows']) == 1),
                            ('multi_row', lambda r: int(r['gold_rows']) > 1),
                            ('at_least_100_rows', lambda r: int(r['gold_rows']) >= 100)]}
(HERE / 'statistics.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n')
fields = ['task_id', 'question', 'domain', 'modality', 'model_phase', 'score', 'passed', 'failure',
          'matched_cols', 'gold_columns', 'prediction_columns', 'gold_rows', 'prediction_rows',
          'latency_minutes', 'timing_source']
with (HERE / 'task_analysis.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    writer.writeheader(); writer.writerows(rows)


def save(fig, name):
    fig.savefig(FIGURES / f'{name}.png', bbox_inches='tight')
    svg_path = FIGURES / f'{name}.svg'
    fig.savefig(svg_path, bbox_inches='tight')
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    plt.close(fig)


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
fig.suptitle('本次公开60题：成绩、组成与两种评分口径', fontsize=17)
ax = axes[0]
values = [summary['reference_baseline_100'], summary['final_course_score_100']]
bars = ax.bar(['教师参考', '当前混合模型'], values, color=[BLUE, GREEN], width=.55)
ax.bar_label(bars, labels=[f'{v:.4f}' for v in values], padding=5)
ax.set_ylim(0, 105); ax.set_ylabel('课程平均分 / 100')
ax.set_title('差值 +3.2640分；不等于受控消融收益', fontsize=11)
ax = axes[1]
ax.barh(['当前课程总分'], [60], color=GREEN, label='36题满分贡献60.00分')
ax.barh(['当前课程总分'], [partial_points], left=[60], color=ORANGE, label=f'15题部分分贡献{partial_points:.2f}分')
ax.axvline(summary['reference_baseline_100'], color=BLUE, linestyle='--', label='教师参考75.6387')
ax.set_xlim(0, 100); ax.set_xlabel('平均分贡献 / 分'); ax.legend(loc='lower center', fontsize=9)
ax.set_title('9题零分没有得分贡献')
ax = axes[2]
bars = ax.bar(['课程满分', '官方通过'], [36, 40], color=[GREEN, BLUE], width=.55)
ax.bar_label(bars, labels=['36/60 (60%)', '40/60 (66.67%)'], padding=5)
ax.set_ylim(0, 60); ax.set_ylabel('题数'); ax.set_title('日期归一差异：4题官方过、课程非满分')
save(fig, '01_score_overview')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout='constrained')
fig.suptitle('评测集分布与表现（描述性分组，不作领域/模型因果排名）', fontsize=15)
for ax, names, data in [
    (axes[0], list(groups), list(groups.values())),
    (axes[1], ['无视频', '含视频'], [group_stats([r for r in rows if r['modality'] == m]) for m in ['无视频', '含视频']])]:
    x = list(range(len(names)))
    b1 = ax.bar([i-.18 for i in x], [g['mean_course_score'] for g in data], .36, label='课程均分 / 100', color=GREEN)
    b2 = ax.bar([i+.18 for i in x], [100*g['official_passed']/g['tasks'] for g in data], .36, label='官方整题通过率 / %', color=BLUE)
    ax.bar_label(b1, fmt='%.1f', padding=3, fontsize=10); ax.bar_label(b2, fmt='%.1f', padding=3, fontsize=10)
    ax.set_xticks(x, [f'{n}\nn={g["tasks"]}' for n,g in zip(names,data)])
    ax.set_ylim(0, 112); ax.legend(loc='upper right', fontsize=9)
save(fig, '02_distribution')

fig, ax = plt.subplots(figsize=(14, 6), layout='constrained')
matrix = [[rows[j*10+i]['score'] for i in range(10)] for j in range(6)]
im = ax.imshow(matrix, cmap='RdYlGn', norm=Normalize(0, 100), aspect='auto')
for i, r in enumerate(rows):
    ax.text(i % 10, i // 10, f'{r["task_id"]}\n{r["score"]:.2f}' + ('\n官方未过' if not r['passed'] else ''),
            ha='center', va='center', fontsize=9, color='#162a32')
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('60题成绩全景：课程分数为色彩，官方未通过单独标注', fontsize=16)
fig.colorbar(im, ax=ax, label='课程分数 / 100', shrink=.8)
save(fig, '03_task_heatmap')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout='constrained')
ax = axes[0]
bars = ax.barh(['值/关系/顺序', '行数不符', '列数不符'], [6, 3, 11], color=[RED, ORANGE, BLUE])
ax.bar_label(bars, padding=4); ax.set_xlim(0, 13); ax.set_xlabel('官方首次失败原因 / 题')
ax.set_title('20题未通过：首次失败，不代表唯一错误')
ax = axes[1]
ax.barh(['损失构成'], [15], color=RED, label='9道零分题：损失15分')
loss = stats['partial_task_loss_points']
ax.barh(['损失构成'], [loss], left=[15], color=ORANGE, label=f'15道部分分题：损失{loss:.4f}分')
ax.set_xlim(0, 25); ax.set_xlabel('相对于100分的平均分损失 / 分')
ax.set_title('当前仍损失21.0973分：零分题占主要部分')
ax.legend(loc='lower center', fontsize=10)
save(fig, '04_errors_and_loss')

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
ax = axes[0]
bars = ax.bar(['plus失败补跑27题', 'max有效补跑26题'], [summary['time']['prior_plus_failed_27_wall_seconds']/60,
              summary['time']['max_repair_26_wall_seconds']/60], color=[RED, GREEN])
ax.bar_label(bars, fmt='%.1f min', padding=4); ax.set_ylim(0, 105); ax.set_ylabel('批次墙钟 / 分钟')
ax.set_xticks([0, 1], ['plus失败补跑\n27题，新增0份', 'max补跑\n26题，新增26份'])
ax.set_title('阶段不同，不能视作等价速度对照')
ax = axes[1]
for i,(counts,label) in enumerate([(summary['api']['prior_failed_plus_27_response_status'],'plus失败阶段'),
                                  (summary['api']['max_repair_26_response_status'],'max补跑阶段')]):
    total = sum(counts.values()); left = 0
    for status,color in [('200',GREEN),('500',RED),('504',ORANGE),('429',BLUE)]:
        value = counts.get(status,0)/total*100
        show_label = (i == 0 and status != '429') or (i == 1 and status == '429')
        ax.barh([label], [value], left=[left], color=color, label=f'HTTP {status}' if show_label else None)
        left += value
ax.set_xlim(0,100);ax.set_xlabel('已记录HTTP响应占比 / %');ax.legend(fontsize=9,loc='lower center')
ax.set_title('接口恢复 ≠ 答案正确')
ax = axes[2]
new_ids = set(json.loads((ROOT/'runs/shared_public60_20261026/protocol.json').read_text())['prior_predictions_preserved'])
for r in rows:
    if r['task_id'] not in new_ids and r['task_id'] != 'task_102':
        ax.scatter(r['latency_minutes'],r['score'], color=GREEN if r['passed'] else RED, alpha=.8)
        if r['latency_minutes']>20 or r['score']==0:
            offset = {'task_298': (-40, 5), 'task_128': (-12, 18), 'task_111': (-40, 5)}.get(r['task_id'], (2, 5))
            ax.annotate(r['task_id'],(r['latency_minutes'],r['score']),xytext=offset,textcoords='offset points',fontsize=8)
ax.set_xlabel('该次任务延迟 / 分钟');ax.set_ylabel('课程分数');ax.set_ylim(-10,115)
ax.set_title('最后26题：耗时长并不保证正确')
save(fig, '05_runtime_and_api')

fig, ax = plt.subplots(figsize=(13, 5), layout='constrained')
ax.set_xlim(0, 13);ax.set_ylim(0, 5);ax.axis('off')
boxes=[(.1,3.1,'task.json + context','数据/视频/文档'),(3.3,3.1,'任务级模型路由','文本plus / 视频max'),
       (6.5,3.1,'视频/文档相关性','投票 + 证据整理'),(9.7,3.1,'solver Agent','SQL + 工具反馈'),
       (9.7,.5,'run_solver 自检','进程/shape/样例'),(6.5,.5,'完成凭据与归档','run_ok + CSV存在'),
       (3.3,.5,'独立课程评分','列签名匹配 + 轻惩罚'),(.1,.5,'官方整表评分','类型/行列数/关系')]
for x,y,title,sub in boxes:
    ax.add_patch(FancyBboxPatch((x,y),2.9,1.15,boxstyle='round,pad=0.05',facecolor='white',edgecolor=BLUE,linewidth=1.5))
    ax.text(x+1.45,y+.76,title,ha='center',fontsize=11)
    ax.text(x+1.45,y+.32,sub,ha='center',fontsize=10,color='#526477')
for (x,y),(xx,yy) in [((3,3.65),(3.25,3.65)),((6.2,3.65),(6.45,3.65)),((9.4,3.65),(9.65,3.65)),
                     ((11.15,3),(11.15,1.7)),((9.65,1.08),(9.4,1.08)),((6.45,1.08),(6.2,1.08))]:
    ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':BLUE,'lw':2})
ax.plot([7.95, 7.95, 1.55], [1.72, 1.95, 1.95], color=BLUE, lw=1.5, linestyle='--')
ax.annotate('', xy=(1.55, 1.72), xytext=(1.55, 1.95),
            arrowprops={'arrowstyle': '->', 'color': BLUE, 'lw': 1.5})
ax.text(6.5,2.35,'关键缺口：执行成功之后，缺少基于题意的强制输出契约与业务一致性验收',ha='center',color=RED,fontsize=12)
ax.text(3.15,.1,'官方评分与课程评分均独立读取预测；不是串行修正答案',ha='center',fontsize=10,color='#526477')
ax.set_title('当前baseline的决策—执行—验收链路',fontsize=17)
save(fig,'06_architecture')

# A single offline HTML file: no CDN, model calls or external assets.
data = json.dumps([{k:r[k] for k in fields} for r in rows],ensure_ascii=False).replace('</','<\\/')
page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Baseline公开60题复盘</title><style>
body{font:15px system-ui,sans-serif;background:#f4f7fb;color:#203146;margin:24px;line-height:1.6}main{max-width:1400px;margin:auto}
h1{font-size:27px}.cards{display:flex;gap:14px;flex-wrap:wrap}.card{background:white;border:1px solid #dae3ee;border-radius:10px;padding:14px;min-width:180px}
.card b{font-size:24px;display:block}select,input{padding:8px;margin:6px;border:1px solid #ccd7e3;border-radius:5px}table{border-collapse:collapse;width:100%;background:white}
td,th{padding:9px;border-bottom:1px solid #e3e8ef;text-align:left;vertical-align:top}th{background:#e9eff6;cursor:pointer}td.q{max-width:620px}
.wrap{overflow:auto}.score{min-width:85px}.bar{height:5px;background:#d9e3ed}.fill{height:100%;background:#26846c}.note{color:#53667c}a{color:#276ca6}
</style><main><h1>当前baseline · 公开60题复盘</h1><p>2026-10-26；33份plus历史输出 + 27份max补跑。领域是人工分组；延迟为最后可取得的单次尝试，不能相加当总耗时。</p>
<div class="cards"><div class="card">课程平均分<b>78.9027 / 100</b></div><div class="card">参考基线差值<b>+3.2640分</b></div><div class="card">官方整题通过<b>40 / 60</b></div><div class="card">满分 / 部分 / 零分<b>36 / 15 / 9</b></div></div>
<p class="note">差值已核实；缺少教师逐题预测，不能把3.264分归因给某一个模块。教师模型配比为32 max / 28 plus，本次为27 max / 33 plus。</p>
<a href="README.md">完整分析</a> · <a href="task_analysis.csv">逐题CSV</a> · <a href="figures/03_task_heatmap.png">成绩全景图</a>
<p><select id="domain"><option value="">所有领域</option><option>证券与基金</option><option>宏观经济</option><option>医疗</option></select>
<select id="modality"><option value="">所有模态</option><option>含视频</option><option>无视频</option></select>
<select id="outcome"><option value="">所有结果</option><option value="zero">课程零分</option><option value="partial">课程部分分</option><option value="full">课程满分</option><option value="failed">官方未通过</option></select>
<input id="search" placeholder="搜索题号或题目" aria-label="搜索题号或题目"></p><p id="summary"></p>
<div class="wrap"><table><thead><tr><th data-key="task_id">题号 ↕</th><th>题目</th><th>领域 / 模态</th><th data-key="score">课程分 ↕</th><th>官方</th><th>首次失败</th><th>匹配列 / 目标列 / 输出列</th><th data-key="latency_minutes">单次延迟 ↕</th></tr></thead><tbody id="body"></tbody></table></div>
<script>const data=__DATA__;let sortKey='task_id',direction=1;const $=id=>document.getElementById(id);
function cell(row,value){const td=document.createElement('td');td.textContent=value;row.appendChild(td);return td}
function render(){const q=$('search').value.toLowerCase(),o=$('outcome').value;let rows=data.filter(r=>(!$('domain').value||r.domain===$('domain').value)&&(!$('modality').value||r.modality===$('modality').value)&&(!q||(r.task_id+' '+r.question).toLowerCase().includes(q))&&(!o||(o==='zero'&&r.score===0)||(o==='partial'&&r.score>0&&r.score<100)||(o==='full'&&r.score===100)||(o==='failed'&&!r.passed)));
rows.sort((a,b)=>direction*(sortKey==='task_id'?Number(a.task_id.slice(5))-Number(b.task_id.slice(5)):a[sortKey]-b[sortKey]));
$('summary').textContent=rows.length?`筛选 ${rows.length} 题 · 课程均分 ${(rows.reduce((s,r)=>s+r.score,0)/rows.length).toFixed(2)} · 官方通过 ${rows.filter(r=>r.passed).length}/${rows.length}`:'没有匹配的题目';
$('body').replaceChildren();for(const r of rows){const tr=document.createElement('tr');cell(tr,r.task_id);cell(tr,r.question).className='q';cell(tr,r.domain+' / '+r.modality);const td=cell(tr,r.score.toFixed(2));td.className='score';const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('div');fill.className='fill';fill.style.width=r.score+'%';bar.appendChild(fill);td.appendChild(bar);cell(tr,r.passed?'通过':'未通过');cell(tr,r.failure||'—');cell(tr,`${r.matched_cols} / ${r.gold_columns} / ${r.prediction_columns}`);cell(tr,r.latency_minutes.toFixed(2)+' min');$('body').appendChild(tr)}}
for(const id of ['domain','modality','outcome','search'])$(id).addEventListener('input',render);document.querySelectorAll('[data-key]').forEach(th=>th.addEventListener('click',()=>{direction=sortKey===th.dataset.key?-direction:1;sortKey=th.dataset.key;render()}));render();</script></main></html>'''
(HERE/'dashboard.html').write_text(page.replace('__DATA__',data))
print('Generated 6 PNG + 6 SVG figures, statistics.json, task_analysis.csv and dashboard.html')
