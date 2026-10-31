"""Local acoustic APPT comparison, including no-prior conditional ablation."""
import copy
import hashlib
import json
import time
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from ppf.appt import LocalAcousticTokenizer,LocalReranker,LocalHierarchicalRefiner,AdaptivePossibilityTree,uncertainty_gate
from ppf.model import CNNBaseline
from ppf.data import synthesize_sequence,SynthesisConfig
from compositional_probes import CFG,load_data

ROOT=Path('appt_results');C=json.loads(Path('experiments/appt_config.json').read_text())

def temperature(logits,labels):
    grid=torch.logspace(-.6,.6,81)
    losses=torch.stack([F.cross_entropy(logits/t,labels) for t in grid])
    return float(grid[losses.argmin()])


def calibration(probs,labels):
    confidence,pred=probs.max(-1);correct=pred==labels;ece=0.
    for i in range(10):
        mask=(confidence>=i/10)&(confidence<(i+1)/10 if i<9 else confidence<=1)
        if mask.any():ece+=float(mask.float().mean()*(confidence[mask].mean()-correct[mask].float().mean()).abs())
    return {'digit_accuracy':float(correct.float().mean()),'digit_ece_10_bins':ece,
            'digit_nll':float(-probs[torch.arange(len(labels)),labels].clamp_min(1e-30).log().mean()),
            'digit_brier':float((probs-F.one_hot(labels,10)).square().sum(-1).mean())}

@torch.inference_mode()
def features(cnn,events):return torch.cat([cnn.encoder(z.flatten(0,1)) for z in events.split(64)])

@torch.inference_mode()
def quantizer_metrics(tok,z):
    out=tok(z);frequency=[torch.bincount(c.flatten(),minlength=64) for c in out['codes']]
    return {'feature_mse_by_prefix':[float((p-z).square().mean()) for p in out['prefixes']],
            'unique_codes':[int((f>0).sum()) for f in frequency],
            'code_frequencies':[f.tolist() for f in frequency]}


def train_models(data,cnn):
    f={s:features(cnn,v[0]) for s,v in data.items()}
    tok=LocalAcousticTokenizer(C['latent_dim'],C['codebook_size']);tok.fit_projection(f['train'])
    with torch.no_grad():z={s:tok.project(v) for s,v in f.items()}
    torch.manual_seed(C['seed']);loader=DataLoader(TensorDataset(z['train']),batch_size=256,shuffle=True)
    for _ in range(C['tokenizer_epochs']):
        tok.train()
        for (x,) in loader:tok(x,cutoff=int(torch.randint(1,4,()).item()))
    tok.freeze();qm={s:quantizer_metrics(tok,v) for s,v in z.items()}
    (ROOT/'tokenizer_metrics.json').write_text(json.dumps(qm,indent=2));torch.save(tok.state_dict(),ROOT/'acoustic_tokenizer.pt')
    models={};histories={}
    for name in ['local','appt','no_prior']:
        torch.manual_seed(C['seed'])
        model=LocalReranker() if name=='local' else LocalHierarchicalRefiner(copy.deepcopy(tok),name!='no_prior')
        initial={k:v.clone() for k,v in tok.state_dict().items()}
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=C['learning_rate'])
        y=data['train'][1].flatten()
        loader=DataLoader(TensorDataset(z['train'],y),batch_size=C['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(C['seed']))
        history=[];start=time.perf_counter()
        for epoch in range(C['epochs']):
            model.train();losses=[]
            for x,labels in loader:
                opt.zero_grad(set_to_none=True)
                if name=='local':loss=F.cross_entropy(model(x),labels)
                else:
                    with torch.no_grad():target=model.tokenizer(x)['codes']
                    out=model(x,differentiable=True)
                    digit=sum(F.cross_entropy(l,labels) for l in out['digit_logits'])/3
                    codes=sum(F.cross_entropy(l.flatten(0,1),c.flatten()) for l,c in zip(out['code_logits'],target))/3
                    loss=digit+C['code_loss_weight']*codes
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
            history.append(float(np.mean(losses)))
            print(name,epoch+1,history[-1],flush=True)
        model.eval();models[name]=model
        histories[name]={'loss':history,'training_seconds':time.perf_counter()-start,
                         'parameters':sum(p.numel() for p in model.parameters()),
                         'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
        if name!='local':assert all(torch.equal(v,initial[k]) for k,v in model.tokenizer.state_dict().items())
        torch.save(model.state_dict(),ROOT/f'{name}.pt')
    temps={}
    with torch.inference_mode():
        y=data['val'][1].flatten();base=cnn.head(f['val'])
        temps['cnn']=temperature(base,y)
        for name,model in models.items():
            logits=model(z['val']) if name=='local' else model(z['val'])['digit_logits'][-1]
            temps[name]=temperature(logits,y)
        calibrated=(base/temps['cnn']).softmax(-1);err=calibrated.argmax(-1)!=y
        threshold=float(torch.quantile(calibrated.max(-1).values[err],C['uncertainty_error_recall_target']))+1e-6 if err.any() else 0.
    diagnostics={}
    with torch.inference_mode():
        for name in ['appt','no_prior']:
            out=models[name](z['test']);target=tok(z['test'])['codes']
            diagnostics[name]={'code_accuracy_by_level':[float((a==b).float().mean()) for a,b in zip(out['codes'],target)],
                'digit_accuracy_by_prefix':[float((l.argmax(-1)==data['test'][1].flatten()).float().mean()) for l in out['digit_logits']]}
    return tok,models,temps,threshold,histories,diagnostics


def scenarios(data):
    x,y,condition=data['test'];rng=np.random.default_rng(C['seed']+1)
    result={'corrupted':(x,y,None)}
    for name in ['noise','interference','partial_erasure']:
        mask=torch.tensor([c==name for c in condition])
        result[name]=(x[mask],y[mask],None)
    clean=[]
    for labels in y:
        wav,meta=synthesize_sequence(''.join(str(int(v)) for v in labels),rng,SynthesisConfig(tone_ms=(180,180),gap_ms=(30,30)),corruption=False)
        clean.append(np.stack([np.pad(wav[e['start_sample']:e['end_sample']],(0,960)) for e in meta['events']]))
    clean=torch.tensor(np.stack(clean));result['clean']=(clean,y,None)
    erased=clean.clone()
    for i in range(len(erased)):erased[i,i%4]=0;erased[i,(i+1)%4]=0
    result['two_erased']=(erased,y,None)
    # Tester creates deliberately wrong 99.5% predictions. The inference
    # system receives no labels, and its audio verifier is unchanged.
    overrides=torch.full((len(x),4,10),float('nan'))
    for i in range(len(x)):
        pos=i%4;overrides[i,pos]=.005/9;overrides[i,pos,(int(y[i,pos])+1)%10]=.995
    result['confident_error']=(x,y,overrides)
    result['confident_error_clean_ood']=(clean,y,overrides)
    return result


def validate_scenarios(variants):
    from ppf.engine import goertzel_predict
    x,y,_=variants['clean']
    correct=sum(goertzel_predict(a.numpy())==int(label)
                for events,labels in zip(x,y) for a,label in zip(events,labels))
    result={'ideal_clean_goertzel_digit_accuracy':correct/(len(x)*4),
            'checked_clean_digits':len(x)*4,
            'exactly_two_zero_windows_per_erased_sequence':bool(
                (variants['two_erased'][0].square().sum(-1)==0).sum(-1).eq(2).all())}
    assert result['ideal_clean_goertzel_digit_accuracy']==1.
    assert result['exactly_two_zero_windows_per_erased_sequence']
    (ROOT/'scenario_validation.json').write_text(json.dumps(result,indent=2))


@torch.inference_mode()
def evaluate_scenario(system,events,labels,override,method,trace_path=None):
    all_probs=[];top1=[];top5=[];counts=[];mass=[];lat=[];recalled=errors=0;reopened=0;refined=0;injected_reopened=0;injected_corrected=0;zero=[]
    # Warmup uses only audio and the same method.
    for _ in range(3):system.predict(events[0],method)
    for i,(x,y) in enumerate(zip(events,labels)):
        replacement=None
        if override is not None:
            with torch.inference_mode():replacement=(system.cnn(x)/system.temperatures['cnn']).softmax(-1)
            mask=torch.isfinite(override[i,:,0]);replacement[mask]=override[i,mask]
        start=time.perf_counter();out=system.predict(x,method,base_override=replacement)
        lat.append((time.perf_counter()-start)*1000)
        p=out['digit_probabilities'];all_probs.append(p)
        top1.append(bool((out['sequences'][0]==y).all()));top5.append(bool((out['sequences']==y).all(-1).any()))
        counts.append(out['branches']);mass.append(out['retained_probability_mass']);reopened+=int(out['reopened'].sum());refined+=out['refined_positions']
        # Error detector denominator is the original calibrated CNN prediction,
        # with the common no-evidence rule and optional injected confidence.
        with torch.inference_mode():base=(system.cnn(x)/system.temperatures['cnn']).softmax(-1)
        if replacement is not None:base=replacement
        base[out['erased']]=.1;wrong=base.argmax(-1)!=y
        errors+=int(wrong.sum());recalled+=int((wrong&out['expanded']).sum())
        if override is not None:
            pos=i%4;injected_reopened+=int(out['expanded'][pos]);injected_corrected+=int(p[pos].argmax()==y[pos])
        if out['erased'].any():zero.append(float((p[out['erased']]-.1).abs().max()))
    probs=torch.cat(all_probs);metrics=calibration(probs,labels.flatten())
    metrics.update(sequence_top1=float(np.mean(top1)),sequence_top5=float(np.mean(top5)),samples=len(events),
        uncertainty_error_recall=recalled/errors if errors else None,cnn_error_digits=errors,
        mean_branches=float(np.mean(counts)),max_branches=int(max(counts)),mean_retained_search_mass=float(np.mean(mass)),
        mean_reopened_positions=reopened/len(events),mean_refined_positions=refined/len(events),
        latency_ms={'p50':float(np.percentile(lat,50)),'p95':float(np.percentile(lat,95))},
        erased_max_deviation_from_uniform=max(zero) if zero else None)
    if override is not None:metrics.update(injected_error_reopening_recall=injected_reopened/len(events),injected_error_digit_recovery=injected_corrected/len(events))
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True,exist_ok=True)
        trace_path.write_text(json.dumps({'top1_correct':top1,'top5_correct':top5,'candidate_leaves':counts},indent=2))
    return metrics


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.manual_seed(C['seed']);torch.use_deterministic_algorithms(True)
    (ROOT/'config.json').write_text(json.dumps(C,indent=2));data=load_data()
    cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.eval()
    torch.save({'model':cnn.state_dict(),'config':cnn.cfg.to_dict()},ROOT/'cnn.pt')
    for p in cnn.parameters():p.requires_grad_(False)
    if '--evaluate-only' in sys.argv:
        saved=json.loads((ROOT/'metrics.json').read_text())
        temps=saved['temperatures'];threshold=saved['confidence_threshold'];history=saved['training'];diag=saved['refinement_diagnostics']
        tok=LocalAcousticTokenizer(C['latent_dim'],C['codebook_size']);tok.load_state_dict(torch.load(ROOT/'acoustic_tokenizer.pt',weights_only=True));tok.freeze()
        models={}
        for name in ['local','appt','no_prior']:
            model=LocalReranker() if name=='local' else LocalHierarchicalRefiner(copy.deepcopy(tok),name!='no_prior')
            model.load_state_dict(torch.load(ROOT/f'{name}.pt',weights_only=True));models[name]=model.eval()
    else:tok,models,temps,threshold,history,diag=train_models(data,cnn)
    results={'config':C,'temperatures':temps,'confidence_threshold':threshold,'training':history,'refinement_diagnostics':diag,'scenarios':{}}
    variants=scenarios(data)
    validate_scenarios(variants)
    # Adaptive methods share a verifier. The ordinary full CNN reference does
    # not execute PCA or the verifier; adaptive CNN isolates search pruning.
    for scenario,(x,y,override) in variants.items():
        rows={}
        for name,method,refiner in [('cnn','cnn',models['appt']),('cnn_adaptive','cnn_adaptive',models['appt']),('local','local',models['appt']),('appt','appt',models['appt']),('no_prior','appt',models['no_prior'])]:
            ts={'cnn':temps['cnn'],'local':temps['local'],'appt':temps['no_prior'] if name=='no_prior' else temps['appt']}
            system=AdaptivePossibilityTree(cnn,tok,models['local'],refiner,ts,threshold,C['audit_confidence_threshold'])
            rows[name]=evaluate_scenario(system,x,y,override,method,ROOT/'traces'/f'{scenario}_{name}.json')
            print(json.dumps({'scenario':scenario,'method':name,'metrics':rows[name]}),flush=True)
        results['scenarios'][scenario]=rows;(ROOT/'metrics.json').write_text(json.dumps(results,indent=2))
    paths=['ppf/appt.py','experiments/appt_config.json','representation_results/cnn.pt']
    (ROOT/'source_hashes.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},indent=2))

if __name__=='__main__':main()
