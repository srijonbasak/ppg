"""Corrected fixed frontend and validation-only shared safety gate, three seeds."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import appt_experiment as ap
from clean_failure_audit import clean_waveforms
from compositional_probes import CFG,load_data
from ppf.model import CNNBaseline
from ppf.safe_gate import RelativeFloorLogMel,BenefitGate,gate_features,SafeAdaptiveTree

ROOT=Path('gate_results');C=json.loads(Path('experiments/safe_gate_config.json').read_text())

@torch.inference_mode()
def proposals(cnn,tok,models,x):
    f=ap.features(cnn,x);z=tok.project(f)
    return {'cnn':cnn.head(f),'local':models['local'](z),'appt':models['appt'](z)['digit_logits'][-1],
            'z':z,'rms':x.flatten(0,1).square().mean(-1).sqrt()}


def fit_gate(cnn,tok,models,data,clean_val,seed):
    # Split by recording, keeping paired clean/corrupted views together.
    order=torch.randperm(len(data['val'][1]),generator=torch.Generator().manual_seed(619))
    half=len(order)//2;fit_ids=order[:half];check_ids=order[half:]
    subsets={}
    for name,ids in [('fit',fit_ids),('check',check_ids)]:
        xx=torch.cat([data['val'][0][ids],clean_val[ids]])
        yy=data['val'][1][ids].repeat(2,1).flatten()
        pred=proposals(cnn,tok,models,xx)
        subsets[name]=(pred,yy)
    pred,y=subsets['fit'];temps={n:ap.temperature(pred[n],y) for n in ['cnn','local','appt']}
    feat=[];targets=[]
    for name in ['local','appt']:
        p=(pred['cnn']/temps['cnn']).softmax(-1);q=(pred[name]/temps[name]).softmax(-1)
        feat.append(gate_features(p,q,pred['rms'],pred['z']))
        ok=p.argmax(-1)==y;candidate=q.argmax(-1)==y
        targets.append(torch.where(ok&(~candidate),0,torch.where((~ok)&candidate,2,1)))
    x=torch.cat(feat).detach().clone();target=torch.cat(targets)
    torch.manual_seed(seed);gate=BenefitGate();gate.mean.copy_(x.mean(0));gate.std.copy_(x.std(0).clamp_min(.01))
    opt=torch.optim.Adam(gate.parameters(),lr=C['gate_learning_rate'])
    for _ in range(C['gate_steps']):
        opt.zero_grad(set_to_none=True);loss=F.cross_entropy(gate(x),target)+C['gate_l2']*gate.linear.weight.square().mean()
        loss.backward();opt.step()
    gate.eval();check,truth=subsets['check']
    with torch.inference_mode():
        p=(check['cnn']/temps['cnn']).softmax(-1);proxy=(check['local']/temps['local']).softmax(-1)
        score=gate.score(gate_features(p,proxy,check['rms'],check['z']));wrong=p.argmax(-1)!=truth
        route_cutoff=float(torch.quantile(score[wrong],1-C['route_error_recall_target']))-1e-6
        old_cutoff=float(torch.quantile(p.max(-1).values[wrong],.9))+1e-6
        route=score>=route_cutoff
        candidate_scores={n:gate.score(gate_features(p,(check[n]/temps[n]).softmax(-1),check['rms'],check['z'])) for n in ['local','appt']}
        records=[]
        for threshold in C['acceptance_thresholds']:
            utility=harm=correct=0
            for name in ['local','appt']:
                q=(check[name]/temps[name]).softmax(-1);accept=route&(candidate_scores[name]>=threshold)
                pred=p.argmax(-1).clone();pred[accept]=q[accept].argmax(-1)
                before=p.argmax(-1)==truth;after=pred==truth
                utility+=int(((~before)&after).sum())-int((before&(~after)).sum())
                harm+=int((before&(~after)).sum());correct+=int(before.sum())
            records.append({'threshold':threshold,'net_corrected_digits':utility,'harm_rate':harm/max(correct,1)})
        eligible=[r for r in records if r['harm_rate']<=C['maximum_correct_to_wrong_rate']]
        chosen=max(eligible,key=lambda r:(r['net_corrected_digits'],-r['harm_rate'],r['threshold']))
        checks={'fit_recordings':fit_ids.tolist(),'check_recordings':check_ids.tolist(),
                'validation_route_error_recall':float(route[wrong].float().mean()),'threshold_trials':records,
                'chosen':chosen,'gate_loss':float(loss.detach())}
    return gate,temps,route_cutoff,chosen['threshold'],old_cutoff,checks

@torch.inference_mode()
def evaluate(system,x,y,override,method,path):
    rows=[];lat=[];probs=[];top1=[];top5=[];corrected=regressed=base_errors=routed_errors=0
    expanded=leaves=refined=0;injected_open=injected_recovery=0
    for _ in range(3):system.predict(x[0],method)
    for i,(a,label) in enumerate(zip(x,y)):
        replacement=None
        if override is not None:
            replacement=(system.cnn(a)/system.temperatures['cnn']).softmax(-1)
            mask=torch.isfinite(override[i,:,0]);replacement[mask]=override[i,mask]
        start=time.perf_counter();out=system.predict(a,method,base_override=replacement)
        lat.append((time.perf_counter()-start)*1000);p=out['digit_probabilities'];probs.append(p)
        before=out['base_probabilities'].argmax(-1)==label;after=p.argmax(-1)==label
        corrected+=int(((~before)&after).sum());regressed+=int((before&(~after)).sum())
        base_errors+=int((~before).sum());routed_errors+=int(((~before)&out['expanded']).sum())
        t1=bool((out['sequences'][0]==label).all());t5=bool((out['sequences']==label).all(-1).any());top1.append(t1);top5.append(t5)
        expanded+=int(out['expanded'].sum());leaves+=out['branches'];refined+=out['refined_positions']
        if override is not None:
            pos=i%4;injected_open+=int(out['expanded'][pos]);injected_recovery+=int(after[pos])
        rows.append({'label':label.tolist(),'top1':t1,'top5':t5,'digit_correct':after.int().tolist()})
    result=ap.calibration(torch.cat(probs),y.flatten())
    result.update(sequence_top1=float(np.mean(top1)),sequence_top5=float(np.mean(top5)),samples=len(x),
                  wrong_to_correct=corrected,correct_to_wrong=regressed,cnn_error_recall=routed_errors/max(base_errors,1),
                  expanded_positions=expanded/len(x),candidate_leaves=leaves/len(x),refined_positions=refined/len(x),
                  latency_ms={'p50':float(np.percentile(lat,50)),'p95':float(np.percentile(lat,95))})
    if override is not None:result.update(injected_reopening=injected_open/len(x),injected_recovery=injected_recovery/len(x))
    path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(rows))
    return result


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    (ROOT/'config.json').write_text(json.dumps(C,indent=2));data=load_data()
    clean_val=clean_waveforms(data['val'][1],619)
    scenarios=ap.scenarios(data)
    selected={s:scenarios[s] for s in ['corrupted','clean','confident_error','two_erased']}
    result={'config':C,'seeds':{}}
    for seed in C['seeds']:
        torch.manual_seed(seed);directory=ROOT/f'seed_{seed}';directory.mkdir(exist_ok=True)
        cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.eval()
        cnn.encoder.frontend=RelativeFloorLogMel(CFG,C['relative_mel_floor'])
        for p in cnn.parameters():p.requires_grad_(False)
        ap.ROOT=directory;ap.C=json.loads(Path('experiments/appt_config.json').read_text());ap.C['seed']=seed
        tok,models,_,_,training,diag=ap.train_models(data,cnn)
        gate,temps,route,accept,old,checks=fit_gate(cnn,tok,models,data,clean_val,seed)
        torch.save(gate.state_dict(),directory/'gate.pt')
        metadata={'temperatures':temps,'route_threshold':route,'accept_threshold':accept,'old_threshold':old,'gate_validation':checks,
                  'training':training,'refinement_diagnostics':diag,'scenarios':{}}
        system=SafeAdaptiveTree(cnn,tok,models['local'],models['appt'],temps,gate,route,accept,old)
        for scenario,(x,y,override) in selected.items():
            rows={}
            for method in ['cnn','cnn_gated','local_old','appt_old','local_gated','appt_gated']:
                rows[method]=evaluate(system,x,y,override,method,directory/'traces'/f'{scenario}_{method}.json')
                print(json.dumps({'seed':seed,'scenario':scenario,'method':method,'metrics':rows[method]}),flush=True)
            metadata['scenarios'][scenario]=rows
        result['seeds'][str(seed)]=metadata
        (ROOT/'metrics.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
