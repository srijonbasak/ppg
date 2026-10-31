"""Gated tokenizer then paired frozen-CNN generator comparison."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from ppf.compositional import CompositionalTokenizer
from ppf.sequence import SequencePPF,mixture_scores
from ppf.model import CNNBaseline
from compositional_probes import ROOT,C,CFG,load_data,original_tokenizer,run_probes
from staged_experiment import tokenizer_metrics

@torch.inference_mode()
def evaluate(model,data):
    model.eval();x,y,conditions=data
    scores=[];digit=[];tfhit=np.zeros(3);ghit=np.zeros(3);counts=np.zeros(3);latency=[]
    for _ in range(3):
        p,_,_=model.generate(x[:1],stochastic=False);mixture_scores(p).topk(5,-1)
    for sample in x:
        begin=time.perf_counter();p,_,_=model.generate(sample[None],stochastic=False)
        s=mixture_scores(p);s.topk(5,-1);latency.append((time.perf_counter()-begin)*1000)
        scores.append(s);digit.append(p[:,0].argmax(-1))
    scores=torch.cat(scores);pred=torch.cat(digit);ids=y@torch.tensor([1000,100,10,1])
    top=scores.topk(5,-1).indices
    def summary(mask):
        return {'samples':int(mask.sum()),'digit_accuracy_by_position':(pred[mask]==y[mask]).float().mean(0).tolist(),
                'sequence_accuracy':float((top[mask,0]==ids[mask]).float().mean()),
                'sequence_top5':float((top[mask]==ids[mask,None]).any(-1).float().mean()),
                'sequence_nll':float(-scores[mask][torch.arange(int(mask.sum())),ids[mask]].clamp_min(1e-30).log().mean())}
    result={'overall':summary(torch.ones(len(x),dtype=torch.bool)),
            'latency_ms':{'p50':float(np.percentile(latency,50)),'p95':float(np.percentile(latency,95))},
            'by_condition':{c:summary(torch.tensor([v==c for v in conditions])) for c in sorted(set(conditions))}}
    for xx,yy in zip(x.split(32),y.split(32)):
        target=model.tokenizer(yy)['codes'];tf=model.teacher_logits(xx,target)
        _,gc,_=model.generate(xx,stochastic=False)
        for l in range(3):
            tfhit[l]+=float((tf[l].argmax(-1)==target[l]).sum());ghit[l]+=float((gc[l]==target[l]).sum());counts[l]+=target[l].numel()
    result['teacher_code_accuracy']=(tfhit/counts).tolist();result['generated_code_accuracy']=(ghit/counts).tolist()
    return result


def train_tokenizer(data):
    torch.manual_seed(C['seed']);tok=CompositionalTokenizer(CFG)
    opt=torch.optim.Adam(tok.parameters(),lr=C['structured_tokenizer_learning_rate'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,C['structured_tokenizer_epochs'],eta_min=.00001)
    loader=DataLoader(TensorDataset(data['train'][1]),batch_size=256,shuffle=True)
    history=[];best=-1;state=None
    for epoch in range(C['structured_tokenizer_epochs']):
        tok.train();losses=[]
        for (y,) in loader:
            opt.zero_grad(set_to_none=True);out=tok(y);out['loss'].backward();torch.nn.utils.clip_grad_norm_(tok.parameters(),1.);opt.step();losses.append(float(out['loss'].detach()))
        metrics=tokenizer_metrics(tok,data['val'][1]);score=metrics['sequence_accuracy_by_level'][-1]
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':metrics})
        if score>best:best=score;state={k:v.clone() for k,v in tok.state_dict().items()};selected=epoch+1
        scheduler.step();print('tokenizer',epoch+1,score,flush=True)
    tok.load_state_dict(state);tok.freeze()
    result={'selected_epoch':selected,'history':history,'validation':tokenizer_metrics(tok,data['val'][1]),'test':tokenizer_metrics(tok,data['test'][1]),'gate_passed':best>=C['tokenizer_gate']}
    (ROOT/'structured_tokenizer_metrics.json').write_text(json.dumps(result,indent=2))
    torch.save(tok.state_dict(),ROOT/'structured_tokenizer.pt')
    if not result['gate_passed']:raise RuntimeError('Structured tokenizer validation gate failed')
    return tok


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    data=load_data();structured=train_tokenizer(data)
    features=torch.load(ROOT/'frozen_audio_features.pt',weights_only=True)
    probes=run_probes(structured,'structured',data,features)
    cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.eval()
    baseline={}
    with torch.inference_mode():
        for split in ['val','test']:
            x,y,conditions=data[split];pred=cnn(x.flatten(0,1)).argmax(-1).reshape(-1,4)
            baseline[split]={'sequence_accuracy':float((pred==y).all(-1).float().mean()),'digit_accuracy_by_position':(pred==y).float().mean(0).tolist()}
    result={'config':C,'cnn_baseline':baseline,'runs':{}}
    for name,tok in [('original',original_tokenizer()),('structured',structured)]:
        torch.manual_seed(C['seed'])
        model=SequencePPF(tok,CFG);model.encoder.load_state_dict(cnn.encoder.state_dict())
        for p in model.encoder.parameters():p.requires_grad_(False)
        before={k:v.clone() for k,v in tok.state_dict().items()}
        encoder_before={k:v.clone() for k,v in model.encoder.state_dict().items()}
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=C['generator_learning_rate'])
        loader=DataLoader(TensorDataset(data['train'][0],data['train'][1]),batch_size=C['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(C['seed']))
        history=[];begin=time.perf_counter()
        for epoch in range(C['generator_epochs']):
            model.train();losses=[]
            for x,y in loader:
                with torch.no_grad():codes=tok(y)['codes']
                opt.zero_grad(set_to_none=True);logits=model.teacher_logits(x,codes)
                loss=sum(F.cross_entropy(l.flatten(0,1),c.flatten()) for l,c in zip(logits,codes))/3
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
            history.append(float(np.mean(losses)));print(name,epoch+1,history[-1],flush=True)
        assert all(torch.equal(v,before[k]) for k,v in tok.state_dict().items())
        assert all(torch.equal(v,encoder_before[k]) for k,v in model.encoder.state_dict().items())
        row={'history':history,'training_seconds':time.perf_counter()-begin,
             'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
             'tokenizer_unchanged':True,'cnn_unchanged':True}
        for split in ['val','test']:row[split]=evaluate(model,data[split])
        torch.save(model.state_dict(),ROOT/f'{name}_generator.pt');result['runs'][name]=row
        (ROOT/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps({'model':name,'test':row['test']}),flush=True)

if __name__=='__main__':main()
