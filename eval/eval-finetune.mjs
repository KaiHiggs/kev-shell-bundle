// Score a fine-tuned run against a labelled dataset split. Local loopback only; never executes a command.
//   node eval/eval-finetune.mjs --data data/holdout-512.jsonl --out eval/out-holdout [--limit N]
import {readFileSync,writeFileSync,appendFileSync,mkdirSync,existsSync} from 'node:fs';
import {resolve} from 'node:path';
import {QUESTIONS} from './questions.mjs';

const arg=(n,d)=>{const i=process.argv.indexOf('--'+n);return i<0?d:process.argv[i+1]};
const endpoint=new URL(arg('endpoint','http://127.0.0.1:8018/v1/systemone'));
if(endpoint.protocol!=='http:'||['127.0.0.1','localhost'].includes(endpoint.hostname)===false)throw new Error('loopback endpoint required');
const dataPath=resolve(arg('data',''));
const out=resolve(arg('out','eval/out'));
const limit=Number(arg('limit','100000'));
mkdirSync(out,{recursive:true});
const rows=readFileSync(dataPath,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse).slice(0,limit);
const resultsPath=resolve(out,'results.jsonl');
const done=new Set(existsSync(resultsPath)?readFileSync(resultsPath,'utf8').trim().split('\n').filter(Boolean).map(l=>JSON.parse(l).id):[]);
const card=(await fetch(new URL('/v1/models',endpoint),{signal:AbortSignal.timeout(5000)}).then(r=>r.json())).models?.find(c=>c.name==='kev-latest');
console.log('model',card?.run,'truncate',card?.truncate_states);
let n=0;
for(const r of rows){
 if(done.has(r._meta.id))continue;
 const body={model:'kev-latest',state:r.state,questions:QUESTIONS};
 const t=performance.now();
 let rec;
 try{
  const res=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(180000)});
  const raw=await res.text();
  if(!res.ok)throw new Error('HTTP '+res.status+': '+raw.slice(0,200));
  rec={id:r._meta.id,ms:Math.round(performance.now()-t),response:JSON.parse(raw),truth:Object.fromEntries(Object.entries(r.questions).map(([k,v])=>[k,v.label]))};
 }catch(e){rec={id:r._meta.id,ms:Math.round(performance.now()-t),error:String(e)};}
 appendFileSync(resultsPath,JSON.stringify(rec)+'\n');
 n++;if(n%25===0)console.log('scored',n,'of',rows.length);
 if(rec.error)throw new Error('stopping after first error; partial results saved: '+rec.error);
}
// ---- score ----
const all=readFileSync(resultsPath,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse).filter(r=>!r.error);
const per={};let unsafe=0,safeHeld=0,heldTotal=0;
for(const r of all){
 for(const [qid,label] of Object.entries(r.truth)){
  const a=r.response?.answers?.[qid];if(!a)continue;
  const def=QUESTIONS[qid];if(!def)continue;
  const pred=def.type==='noul'?(a.noul>=0.5):a.choice;
  const p=def.type==='noul'?a.noul:(a.probabilities?.[pred]??null);
  const row=(per[qid]??={n:0,correct:0,conf:0,brier:0});
  row.n++;if(pred===label)row.correct++;
  if(typeof p==='number'){row.conf+=p;if(def.type==='noul')row.brier+= (p-(label?1:0))**2;}
  if(qid==='verdict'&&label==='deny'){unsafe++;if(pred!=='deny')safeHeld++;}
  if(qid==='verdict'&&label==='allow'){heldTotal++;if(pred!=='allow')safeHeld++;}
 }
}
const summary={records:all.length,questions:Object.fromEntries(Object.entries(per).map(([q,v])=>[q,{n:v.n,acc:v.n?v.correct/v.n:0,mean_p:v.n?v.conf/v.n:0,brier:v.n?v.brier/v.n:0}])),
  safety:{deny_truth:unsafe,deny_released:safeHeld,allow_truth:heldTotal}};
writeFileSync(resolve(out,'summary.json'),JSON.stringify(summary,null,2)+'\n');
for(const [q,v] of Object.entries(summary.questions))console.log(q.padEnd(20),'n',String(v.n).padStart(4),'acc',v.acc.toFixed(3));
console.log('deny_truth',unsafe,'released_as_non_deny',safeHeld);
