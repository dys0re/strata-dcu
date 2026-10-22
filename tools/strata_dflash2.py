"""Run a Strata --serve command with the experimental DFlash2 IPC drafter.

Example: python tools/strata_dflash2.py --checkpoint /persistent/draft --model /persistent/target --stats-file /persistent/run/worker.json -- /path/to/new/strata ... --serve
The target command must disable native MTP, prefix/conversation caches, suffix and
lookup drafting; use --spec 4 --mtp-max-t 4 for three proposals per verify window.
"""
import argparse,json,os,socket,subprocess,sys,threading,time,signal
from pathlib import Path
from dflash2_worker import Dflash2Worker

def main():
 ap=argparse.ArgumentParser(__doc__);ap.add_argument('--checkpoint',required=True);ap.add_argument('--model',required=True);ap.add_argument('--stats-file',required=True);ap.add_argument('--graph',action='store_true');ap.add_argument('--adaptive',action='store_true');ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
 cmd=a.command[1:] if a.command[:1]==['--'] else a.command
 if not cmd:ap.error('engine command required after --')
 stats=Path(a.stats_file);stats.parent.mkdir(parents=True,exist_ok=True)
 error=stats.with_suffix('.error.txt');error.unlink(missing_ok=True)
 os.environ['STRATA_DFLASH_GRAPH']='1' if a.graph else '0'
 os.environ['STRATA_DFLASH_ADAPTIVE']='1' if a.adaptive else '0'
 started=time.monotonic();worker=Dflash2Worker(a.checkpoint,a.model);load_s=time.monotonic()-started
 local,child=socket.socketpair();env=dict(os.environ,STRATA_DFLASH_FD=str(child.fileno()),STRATA_MAX_WEIGHT_VRAM_FRACTION='0.5')
 thread=threading.Thread(target=worker.serve,args=(local,error),daemon=True);thread.start()
 print('strata DFlash2 worker: loaded',worker.weight_bytes,'GPU weight bytes in',round(load_s,2),'s',file=sys.stderr,flush=True)
 p=None
 def terminate(signum,frame):raise KeyboardInterrupt
 signal.signal(signal.SIGTERM,terminate)
 try:
  p=subprocess.Popen(cmd,env=env,pass_fds=(child.fileno(),));child.close();code=p.wait()
 finally:
  child.close()
  if p is not None and p.poll() is None:p.terminate();p.wait(timeout=20)
  thread.join(timeout=10)
  if thread.is_alive():local.close()
  stats.write_text(json.dumps({'checkpoint':str(Path(a.checkpoint).resolve()),'model':str(Path(a.model).resolve()),'command':cmd,'weight_bytes':worker.weight_bytes,'load_s':load_s,'graphs':len(worker.graphs),'adaptive':worker.adaptive,'budget_history':worker.budget_history+[worker.budget.report()],'totals':worker.totals,'timings':getattr(worker,'timings',[]),'worker_error':error.read_text() if error.exists() else None,'engine_exit':p.returncode if p else None},indent=2)+'\n')
 if error.exists():return 1
 return code
if __name__=='__main__':raise SystemExit(main())
