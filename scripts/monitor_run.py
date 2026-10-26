"""Read-only public60 progress report; no gold answers or API requests."""
import json,re,os,time,sys
from pathlib import Path
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path(__file__).resolve().parents[1];run=Path(sys.argv[1]).resolve()
ids=json.loads((root/'data/public_reference_tasks.json').read_text())['task_ids']
session=json.loads((run/'current_session.json').read_text()) if (run/'current_session.json').exists() else {}
if (run/'console.log').exists():
 with (run/'console.log').open('rb') as console:
  console.seek(session.get('console_start_byte',0));text=console.read().decode(errors='replace')
else:text=''
plain=re.sub(r'\x1b\[[0-9;]*m','',text)
started=re.findall(r'Now process (task_\d+)',plain)
receipts=[]
for t in ids:
 p=run/'predictions'/t/'completion.json'
 if p.is_file():
  try:
   if json.loads(p.read_text()).get('success') is True:receipts.append(t)
  except ValueError:pass
attempted=[t for t in ids if (run/'temp/logs'/t/'timing.json').is_file()]
metrics=[]
p=run/'temp/logs/model_requests.jsonl'
if p.is_file():
 for line in p.read_text().splitlines():
  try:metrics.append(json.loads(line))
  except ValueError:pass
status=Counter(str(x['status']) for x in metrics)
usage=Counter()
for r in metrics:
 for k,v in r.get('usage',{}).items():
  if isinstance(v,(int,float)):usage[k]+=v
lines=[x for x in plain.splitlines() if '[fanout]' in x or '⏱' in x or 'agent round #' in x or 'doc_relevance ok' in x]
now=datetime.now(ZoneInfo('Asia/Shanghai'));stamp=(run/'versions.txt').read_text().splitlines()[0]
elapsed_until=now
if (run/'exit-code.txt').exists() and session.get('finished_at'):
 try:elapsed_until=datetime.fromisoformat(session['finished_at'])
 except ValueError:pass
try:elapsed=(elapsed_until-datetime.fromisoformat(stamp)).total_seconds()
except ValueError:elapsed=None
resume=json.loads((run/'resume.json').read_text()) if (run/'resume.json').exists() else {}
report={'reused_successful':len(resume.get('reused_tasks',[])),'updated_at':now.isoformat(),'run_dir':str(run),'expected':60,'attempted':len(attempted),'successful':len(receipts),'prediction_files':sum((run/'predictions'/t/'prediction.csv').is_file() for t in ids),'current_task':started[-1] if started else None,'active_tasks':list(dict.fromkeys(t for t in started if t not in attempted)),'successful_tasks':receipts,'finished_without_success':[t for t in attempted if t not in receipts],'elapsed_minutes':round(elapsed/60,1) if elapsed else None,'responses':len(metrics),'http_status':dict(status),'provider_usage':dict(usage),'last_progress':lines[-1][-350:] if lines else None,'ended':(run/'exit-code.txt').is_file()}
if report['ended']:report['exit_code']=(run/'exit-code.txt').read_text().strip()

current_metrics=[]
if session.get('started_at'):
 try:
  session_start=datetime.fromisoformat(session['started_at'])
  current_metrics=[row for row in metrics if datetime.fromisoformat(row['timestamp'])>=session_start]
  report['elapsed_minutes_this_session']=round((elapsed_until-session_start).total_seconds()/60,1)
  report['responses_this_session']=len(current_metrics)
  report['http_status_this_session']=dict(Counter(str(row['status']) for row in current_metrics))
  report['successful_this_session']=len(set(receipts)-set(session.get('reused_tasks',[])))
  report['attempted_this_session']=len(set(attempted)&set(session.get('pending_tasks',[])))
 except (ValueError,KeyError,TypeError):pass

try:
 mem={line.split(':',1)[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>1 and line.split()[1].isdigit()}
 report['resources']={'memory_available_mib':round(mem['MemAvailable']/1024), 'swap_free_mib':round(mem['SwapFree']/1024)}
 launcher=int(session.get('launcher_pid',run.name.rsplit('_',1)[-1]));pending=[launcher];seen=set();agents=[]
 while pending:
  pid=pending.pop()
  if pid in seen:continue
  seen.add(pid)
  proc=Path('/proc')/str(pid)
  try:
   cmd=(proc/'cmdline').read_bytes()
   if b'zz_agent_v2.py' in cmd:
    status=(proc/'status').read_text().splitlines()
    rss=next((int(line.split()[1]) for line in status if line.startswith('VmRSS:')),0)
    agents.append((rss,pid))
   pending.extend(int(x) for x in (proc/f'task/{pid}/children').read_text().split())
  except (OSError,ValueError):pass
 if agents:
  rss,pid=max(agents);report['resources'].update(agent_pid=pid,agent_rss_mib=round(rss/1024))
except (OSError,ValueError,KeyError):pass
report['process_running']=bool(report.get('resources',{}).get('agent_pid'))
try:
 launcher_command=(Path('/proc')/str(session.get('launcher_pid',run.name.rsplit('_',1)[-1]))/'cmdline').read_bytes()
 launcher_running=session.get('launcher_marker','run_tasks.sh').encode() in launcher_command
except OSError:launcher_running=False
report['runtime_state']=('finished' if report['ended'] else
                         'running' if report['process_running'] else
                         'initializing' if launcher_running else
                         'interrupted_without_exit_record')
if report['ended'] or report['runtime_state']=='interrupted_without_exit_record':
 report['unfinished_tasks_from_log']=report['active_tasks']
 report['active_tasks']=[]
if report['ended']:
 report['last_started_task']=report['current_task']
 report['current_task']=None
budget_file=run/'temp/logs/request_budget.json'
budget_is_current=budget_file.exists()
if budget_is_current and session.get('started_at'):
 try:budget_is_current=budget_file.stat().st_mtime>=datetime.fromisoformat(session['started_at']).timestamp()
 except (OSError,ValueError):budget_is_current=False
if budget_is_current:
 try:report['request_budget']=json.loads(budget_file.read_text())
 except (OSError,ValueError):pass
else:
 try:
  configured=int(json.loads((run/'run_manifest.json').read_text())['identity']['configuration']['MODEL_MAX_CONCURRENCY'])
  report['request_budget']={'configured_ceiling':configured,'effective_limit':configured,'state':'initial_current_session_budget'}
 except (OSError,ValueError,KeyError,TypeError):pass
if session.get('started_at'):
 report['models_this_session']=dict(Counter(str(row.get('requested_model')) for row in current_metrics))
promotion_file=run/'pending-promotion.json'
if promotion_file.exists():
 try:report['pending_promotion']=json.loads(promotion_file.read_text())
 except (OSError,ValueError):pass
temp=run/f'progress.{os.getpid()}.tmp'
temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
temp.replace(run/'progress.json')
print(json.dumps({k:v for k,v in report.items() if k not in {'run_dir','successful_tasks','provider_usage'}},ensure_ascii=False))
