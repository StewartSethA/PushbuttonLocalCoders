#!/usr/bin/env python3
import json,os,random,secrets,sys,textwrap
from pathlib import Path
R=Path(os.environ.get('LAB_ROOT',str(Path.cwd()/'.cpu-llama-lab')));Q=R/'quality';inp=Q/'human-responses.jsonl';scores=Q/'human-scores.jsonl';blind=os.environ.get('JUDGE_BLIND','1')!='0'
if not inp.exists():raise SystemExit('No human-responses.jsonl; run quality')
items=[json.loads(x) for x in inp.read_text().splitlines() if x.strip()]; keys=sorted({x['candidate_key'] for x in items}); mf=Q/'blind-map.json'
if mf.exists(): aliases=json.loads(mf.read_text())
else:
 aliases={};used=set()
 for k in keys:
  while True:
   a=f'Candidate-{secrets.randbelow(9000)+1000}'
   if a not in used:break
  aliases[k]=a;used.add(a)
 mf.write_text(json.dumps(aliases,indent=2))
sf=Q/'judge-session.json'
if sf.exists():seed=json.loads(sf.read_text())['seed']
else:seed=secrets.randbits(64);sf.write_text(json.dumps({'seed':seed}))
r=random.Random(seed);r.shuffle(items);done={}
if scores.exists():
 for l in scores.read_text().splitlines():
  try:x=json.loads(l);done[x['response_id']]=x
  except:pass
for i,x in enumerate(items):
 if x['response_id'] in done:continue
 print('\033[2J\033[H',end='');print(f'{i+1}/{len(items)} blind={blind} scored={len(done)}');print('='*100);print(x['prompt']);print('-'*100);print(aliases[x['candidate_key']] if blind else x['candidate_key']);print('-'*100);print(x['response']);print('='*100)
 while True:
  a=input('0 wrong | 1 poor | 2 usable | 3 good | 4 excellent | s skip | u reveal | q quit > ').strip().lower()
  if a in '01234' and len(a)==1:
   rec={'response_id':x['response_id'],'candidate_key':x['candidate_key'],'alias':aliases[x['candidate_key']],'prompt_id':x['prompt_id'],'score':int(a)}
   with open(scores,'a') as f:f.write(json.dumps(rec)+'\n');done[x['response_id']]=rec
   break
  if a=='s':break
  if a=='u':print('REAL SOURCE:',x['candidate_key'])
  if a=='q':raise SystemExit
