import os,subprocess,time
from pathlib import Path
pid=46050
while True:
 try:
  state=Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[0]
  if state=='Z':break
 except FileNotFoundError:break
 time.sleep(5)
print('MATRIX PROCESS EXITED',flush=True)
r=subprocess.run(['python3','-u','/root/private_data/strata-dcu/lookup-compat-20261023/run_suite.py'])
raise SystemExit(r.returncode)
