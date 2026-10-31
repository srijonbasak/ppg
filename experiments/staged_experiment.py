"""Reproducible gated tokenizer training then existing-CNN versus staged PPF.
Run PYTHONPATH=. .venv/bin/python experiments/staged_experiment.py --stage tokenizer
or --stage audio after the saved tokenizer has passed the validation gate.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from ppf.config import ModelConfig
from ppf.model import CNNBaseline
from ppf.sequence import CleanLabelTokenizer, SequencePPF, mixture_scores
from ppf.data import synthesize_sequence, SynthesisConfig

ROOT=Path('representation_results')
CONFIG=Path('experiments/staged_config.json')
C=json.loads(CONFIG.read_text())
CFG=ModelConfig()

def setup():
    ROOT.mkdir(exist_ok=True)
    torch.set_num_threads(C['threads'])
    torch.manual_seed(C['seed']); np.random.seed(C['seed']); random.seed(C['seed'])
    torch.use_deterministic_algorithms(True)
    (ROOT/'config.json').write_text(json.dumps(C,indent=2))

def label_splits():
    universe=torch.cartesian_prod(*[torch.arange(10)]*4)
    order=torch.randperm(10000,generator=torch.Generator().manual_seed(C['seed']))
    return universe[order[:8000]],universe[order[8000:9000]],universe[order[9000:]]

@torch.inference_mode()
def tokenizer_metrics(model,labels):
    model.eval(); digit=np.zeros(3); seq=np.zeros(3); qe=np.zeros(3)
    freq=[torch.zeros(CFG.codebook_size) for _ in range(3)]
    for (batch,) in DataLoader(TensorDataset(labels),batch_size=256):
        out=model(batch)
        for l in range(3):
            ok=out['logits'][l].argmax(-1)==batch
            digit[l]+=float(ok.sum()); seq[l]+=float(ok.all(-1).sum())
            qe[l]+=float(out['quantization_error'][l])*len(batch)
            freq[l]+=torch.bincount(out['codes'][l].flatten(),minlength=CFG.codebook_size)
    usage=[]
    for counts,book in zip(freq,model.books):
        p=counts/counts.sum()
        usage.append({'counts':counts.tolist(),'unique_codes':int((counts>0).sum()),
                      'perplexity':float((-(p*p.clamp_min(1e-20).log()).sum()).exp()),
                      'dead_code_replacements':int(book.replacements),
                      'codebook_gradients':'EMA buffers; no optimizer gradients'})
    return {'samples':len(labels),'digit_accuracy_by_level':(digit/(4*len(labels))).tolist(),
            'sequence_accuracy_by_level':(seq/len(labels)).tolist(),
            'quantization_mse_by_level':(qe/len(labels)).tolist(),'usage':usage}

def tokenizer_stage():
    train,val,test=label_splits()
    model=CleanLabelTokenizer(CFG)
    opt=torch.optim.Adam(model.parameters(),lr=C['tokenizer_learning_rate'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,C['tokenizer_epochs'],eta_min=.00001)
    loader=DataLoader(TensorDataset(train),batch_size=C['tokenizer_batch_size'],shuffle=True)
    history=[]; start=time.perf_counter(); best=-1.; best_state=None; selected_epoch=None
    for epoch in range(C['tokenizer_epochs']):
        model.train(); loss_sum=grad_sum=0.; seen=0
        for (labels,) in loader:
            opt.zero_grad(set_to_none=True); out=model(labels); out['loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); loss_sum+=float(out['loss'].detach())*len(labels); grad_sum+=float(norm);seen+=len(labels)
        metrics=tokenizer_metrics(model,val)
        row={'epoch':epoch+1,'loss':loss_sum/seen,'mean_gradient_norm':grad_sum/len(loader),
             'validation':metrics}
        history.append(row)
        if metrics['sequence_accuracy_by_level'][-1]>best:
            best=metrics['sequence_accuracy_by_level'][-1]
            best_state={k:v.clone() for k,v in model.state_dict().items()}
            selected_epoch=epoch+1
        scheduler.step()
        print(json.dumps({'epoch':epoch+1,'loss':row['loss'],'val_sequence':metrics['sequence_accuracy_by_level']}),flush=True)
    model.load_state_dict(best_state)
    metrics=tokenizer_metrics(model,val)
    passed=metrics['sequence_accuracy_by_level'][-1]>=C['reconstruction_gate']
    result={'config':C,'architecture':CFG.to_dict(),'stage':'clean_label_tokenizer',
            'validation_gate_passed':passed,'gate_threshold':C['reconstruction_gate'],'selected_epoch':selected_epoch,
            'training_seconds':time.perf_counter()-start,'history':history,
            'validation':metrics,'test':tokenizer_metrics(model,test)}
    torch.save({'model':model.state_dict(),'config':CFG.to_dict(),'gate_passed':passed},ROOT/'tokenizer.pt')
    (ROOT/'tokenizer_metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'gate_passed':passed,'test':result['test']}),flush=True)

# Four oracle-separated event windows. All corruption is applied to waveforms
# before the frontend. Partial erasure removes only the center 60% of one tone;
# surviving audio is deliberately informative. It is not complete erasure.
def make_audio():
    splits={}; records=[]; hashes=set()
    for sidx,(split,count) in enumerate([('train',C['audio_train_sequences']),('val',C['audio_val_sequences']),('test',C['audio_test_sequences'])]):
        rng=np.random.default_rng(C['seed']+100+sidx)
        audios=[]; labels=[]; conditions=[]
        for j in range(count):
            condition=C['conditions'][j%len(C['conditions'])]
            digits=rng.integers(0,10,4)
            audio,meta=synthesize_sequence(''.join(map(str,digits)),rng,SynthesisConfig(tone_ms=(180,180),gap_ms=(30,30)))
            events=[]
            erased_pos=int(rng.integers(4))
            for pos,e in enumerate(meta['events']):
                tone=audio[e['start_sample']:e['end_sample']].copy()
                if condition=='noise':
                    # Fixed noise amplitude, independent of digit identity.
                    tone+=rng.normal(0,.65,len(tone)).astype(np.float32)
                elif condition=='interference':
                    t=np.arange(len(tone))/8000
                    freq=rng.choice([697,770,852,941,1209,1336,1477])
                    tone=.15*tone+(.65*np.sin(2*np.pi*freq*t+rng.uniform(0,2*np.pi))).astype(np.float32)
                    tone+=rng.normal(0,.10,len(tone)).astype(np.float32)
                elif condition=='partial_erasure' and pos==erased_pos:
                    lo=int(.2*len(tone));hi=int(.8*len(tone))
                    tone[lo:hi]=rng.normal(0,.15,hi-lo)
                    tone*=.15
                    tone+=rng.normal(0,.08,len(tone)).astype(np.float32)
                events.append(np.pad(tone,(0,2400-len(tone))))
            events=np.stack(events).astype(np.float32)
            digest=hashlib.sha256(events.tobytes()).hexdigest()
            if digest in hashes: raise AssertionError('Duplicate audio across experiment')
            hashes.add(digest)
            audios.append(events); labels.append(digits); conditions.append(condition)
            records.append({'split':split,'index':j,'digits':digits.tolist(),'condition':condition,'sha256':digest})
        splits[split]=(torch.tensor(np.stack(audios)),torch.tensor(np.stack(labels)),conditions)
    (ROOT/'audio_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    torch.save(splits,ROOT/'audio_data.pt')
    return splits

def seq_metrics(scores,truth):
    universe=torch.cartesian_prod(*[torch.arange(10)]*4)
    idx=truth@torch.tensor([1000,100,10,1])
    top=scores.topk(5,-1).indices
    confidence=scores.max(-1).values
    hits=top[:,0]==idx
    ece=0.
    for j in range(10):
        mask=(confidence>=j/10)&(confidence<(j+1)/10 if j<9 else confidence<=1)
        if mask.any():ece+=float(mask.float().mean()*(confidence[mask].mean()-hits[mask].float().mean()).abs())
    return {'sequence_top1':float((top[:,0]==idx).float().mean()),
            'sequence_top5':float((top==idx[:,None]).any(-1).float().mean()),
            'sequence_nll':float(-scores[torch.arange(len(idx)),idx].clamp_min(1e-30).log().mean()),
            'digit_accuracy':float((universe[top[:,0]]==truth).float().mean()),
            'sequence_ece_10_bins':ece,
            'sequence_brier':float((scores.square().sum(-1)-2*scores[torch.arange(len(idx)),idx]+1).mean())}

@torch.inference_mode()
def evaluate(model,data,ppf=False):
    model.eval();audio,labels,conditions=data
    scores=[];latencies=[];teacher_hits=np.zeros(3);gen_hits=np.zeros(3);totals=np.zeros(3)
    for _ in range(3):
        if ppf:model.sequence_probabilities(audio[:1],C['hierarchy_samples'])
        else:mixture_scores(model(audio[:1].reshape(-1,2400)).softmax(-1).reshape(1,1,4,10))
    for i in range(len(audio)):
        start=time.perf_counter()
        if ppf: score=model.sequence_probabilities(audio[i:i+1],C['hierarchy_samples'])
        else: score=mixture_scores(model(audio[i]).softmax(-1)[None,None])
        score.topk(5,-1)
        latencies.append((time.perf_counter()-start)*1000);scores.append(score)
    scores=torch.cat(scores)
    result={'overall':seq_metrics(scores,labels),'by_condition':{},
            'latency_ms':{'p50':float(np.percentile(latencies,50)),'p95':float(np.percentile(latencies,95))},
            'parameters':sum(p.numel() for p in model.parameters()),'gpu_memory_bytes':None}
    for cond in C['conditions']:
        mask=torch.tensor([c==cond for c in conditions])
        result['by_condition'][cond]=seq_metrics(scores[mask],labels[mask])
    if ppf:
        for begin in range(0,len(audio),32):
            x=audio[begin:begin+32]; y=labels[begin:begin+32]
            codes=model.tokenizer(y)['codes']
            tf=model.teacher_logits(x,codes)
            _,generated,_=model.generate(x,stochastic=False)
            for l in range(3):
                teacher_hits[l]+=int((tf[l].argmax(-1)==codes[l]).sum())
                gen_hits[l]+=int((generated[l]==codes[l]).sum())
                totals[l]+=codes[l].numel()
        result['teacher_forced_code_accuracy_by_level']=(teacher_hits/totals).tolist()
        result['generated_code_accuracy_by_level']=(gen_hits/totals).tolist()
    return result

def audio_stage():
    payload=torch.load(ROOT/'tokenizer.pt',weights_only=True)
    if not payload['gate_passed']: raise RuntimeError('Tokenizer validation gate failed; audio stage blocked')
    tokenizer=CleanLabelTokenizer(CFG);tokenizer.load_state_dict(payload['model']);tokenizer.freeze()
    data=make_audio()
    train_audio,train_labels,_=data['train']
    cnn=CNNBaseline(CFG)
    loader=DataLoader(TensorDataset(train_audio,train_labels),batch_size=C['audio_batch_size'],shuffle=True)
    opt=torch.optim.AdamW(cnn.parameters(),lr=C['audio_learning_rate'])
    history={'cnn':[],'ppf':[]}; start=time.perf_counter()
    for epoch in range(C['audio_epochs']):
        cnn.train();losses=[]
        for x,y in loader:
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(cnn(x.flatten(0,1)),y.flatten());loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        history['cnn'].append(float(np.mean(losses))); print('CNN',epoch+1,history['cnn'][-1],flush=True)
    cnn_seconds=time.perf_counter()-start
    torch.save(cnn.state_dict(),ROOT/'cnn.pt')
    ppf=SequencePPF(tokenizer,CFG)
    # Explicit warm start using the same trained baseline encoder. This gives
    # PPF extra training compute; it is not a matched-compute comparison.
    ppf.encoder.load_state_dict(cnn.encoder.state_dict())
    opt=torch.optim.AdamW([p for p in ppf.parameters() if p.requires_grad],lr=C['audio_learning_rate'])
    before={k:v.clone() for k,v in tokenizer.state_dict().items()}
    start=time.perf_counter()
    for epoch in range(C['audio_epochs']):
        ppf.train();losses=[]
        for x,y in loader:
            with torch.no_grad():codes=tokenizer(y)['codes']
            opt.zero_grad(set_to_none=True);logits=ppf.teacher_logits(x,codes)
            loss=sum(F.cross_entropy(l.flatten(0,1),c.flatten()) for l,c in zip(logits,codes))/3
            loss.backward();torch.nn.utils.clip_grad_norm_(ppf.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        history['ppf'].append(float(np.mean(losses))); print('PPF',epoch+1,history['ppf'][-1],flush=True)
    ppf_seconds=time.perf_counter()-start
    assert all(torch.equal(before[k],v) for k,v in tokenizer.state_dict().items())
    torch.save(ppf.state_dict(),ROOT/'ppf.pt')
    torch.manual_seed(C['seed']+999)
    result={'config':C,'scope':'single training seed; oracle event boundaries; PPF warm-started from baseline encoder',
            'history':history,'training_seconds':{'cnn':cnn_seconds,'ppf_additional':ppf_seconds},
            'tokenizer_unchanged':True,'duplicate_audio':0,
            'validation':{'cnn':evaluate(cnn,data['val']),'ppf':evaluate(ppf,data['val'],True)},
            'test':{'cnn':evaluate(cnn,data['test']),'ppf':evaluate(ppf,data['test'],True)}}
    (ROOT/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['test'],indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['tokenizer','audio'],required=True);a=p.parse_args()
    setup()
    if a.stage=='tokenizer':tokenizer_stage()
    else:audio_stage()
