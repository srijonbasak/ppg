import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import appt_experiment as ap
from compositional_probes import CFG,load_data
from clean_failure_audit import clean_waveforms
from ppf.model import CNNBaseline
from ppf.appt import LocalAcousticTokenizer,LocalReranker,LocalHierarchicalRefiner,possibility_tree
from ppf.sequence import EMACodebook
from ppf.safe_gate import RelativeFloorLogMel,BenefitGate,SafeAdaptiveTree
from ppf.hierarchy_study import HierarchyStudyVariant

ROOT=Path('hierarchy_results');C=json.loads(Path('experiments/hierarchy_config.json').read_text())


def reference(seed,cnn):
    folder=Path('gate_results')/f'seed_{seed}';meta=json.loads(Path('gate_results/metrics.json').read_text())['seeds'][str(seed)]
    tok=LocalAcousticTokenizer();tok.load_state_dict(torch.load(folder/'acoustic_tokenizer.pt',weights_only=True));tok.freeze()
    local=LocalReranker();local.load_state_dict(torch.load(folder/'local.pt',weights_only=True))
    ref=LocalHierarchicalRefiner(tok);ref.load_state_dict(torch.load(folder/'appt.pt',weights_only=True))
    gate=BenefitGate();gate.load_state_dict(torch.load(folder/'gate.pt',weights_only=True))
    system=SafeAdaptiveTree(cnn,tok,local,ref,meta['temperatures'],gate,meta['route_threshold'],meta['accept_threshold'],meta['old_threshold'])
    return tok,meta,system

@torch.inference_mode()
def freeze_policy(system,x,y,path):
    # Labels are saved for evaluation only, never passed to predict.
    rows=[];base=[];route=[];accept=[]
    for events,label in zip(x,y):
        out=system.predict(events,'local_gated')
        p=out['base_probabilities'];base.append(p);route.append(out['expanded']);accept.append(out['accepted'])
        choices=[list(range(10)) if out['expanded'][j] else [int(p[j].argmax())] for j in range(4)]
        rows.append({'choices':choices,'accepted_positions':out['accepted'].tolist()})
    text=json.dumps(rows,sort_keys=True);path.write_text(text)
    return {'base':torch.stack(base),'route':torch.stack(route),'accept':torch.stack(accept),
            'hash':hashlib.sha256(text.encode()).hexdigest()}


def training_flops(model,z,y,codes):
    model.train();model.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
        model.loss(z,y,codes).backward()
    count=sum(e.flops for e in prof.key_averages());model.zero_grad(set_to_none=True)
    if count<=0:raise RuntimeError('No training FLOPs captured')
    return count

@torch.inference_mode()
def evaluate(model,cnn,tok,x,y,policy,temp,path,reference_system=None):
    z=tok.project(ap.features(cnn,x));r=model.representation(z)
    # A linear decoder of each 128-dimensional representation tests accessibility,
    # not whether a deterministic function creates new acoustic observations.
    scores=[];probs=[];lat=[];changed_good=changed_bad=0;top1=[];top5=[]
    for _ in range(3):
        _=model(z[:4])
    for i,events in enumerate(x):
        start=time.perf_counter()
        f=cnn.encoder(events);zz=tok.project(f)
        if reference_system is not None:
            from ppf.safe_gate import gate_features
            s=reference_system
            base=(cnn.head(f)/s.temperatures['cnn']).softmax(-1)
            proxy=(s.verifier(zz)/s.temperatures['local']).softmax(-1)
            rms=events.square().mean(-1).sqrt()
            route=s.gate.score(gate_features(base,proxy,rms,zz))>=s.route_cutoff
            accepted=route&(s.gate.score(gate_features(base,proxy,rms,zz))>=s.accept_cutoff)
            assert torch.equal(route,policy['route'][i])
            assert torch.equal(accepted,policy['accept'][i])
        p=policy['base'][i].clone();chosen=policy['accept'][i]
        if chosen.any():p[chosen]=(model(zz[chosen])/temp).softmax(-1)
        erased=events.square().mean(-1)<=1e-12;p[erased]=.1
        tree=possibility_tree(p,policy['route'][i],5)
        lat.append((time.perf_counter()-start)*1000)
        before=policy['base'][i].argmax(-1)==y[i];after=p.argmax(-1)==y[i]
        changed_good+=int(((~before)&after).sum());changed_bad+=int((before&(~after)).sum())
        top1.append(bool((tree['sequences'][0]==y[i]).all()));top5.append(bool((tree['sequences']==y[i]).all(-1).any()))
        probs.append(p)
    result=ap.calibration(torch.cat(probs),y.flatten())
    result.update(sequence_top1=float(np.mean(top1)),sequence_top5=float(np.mean(top5)),
                  wrong_to_correct=changed_good,correct_to_wrong=changed_bad,
                  expanded_positions=float(policy['route'].float().sum(-1).mean()),
                  accepted_positions=float(policy['accept'].float().sum(-1).mean()),
                  candidate_leaves=float((10**policy['route'].sum(-1)).float().mean()),
                  latency_replay_ms={'p50':float(np.percentile(lat,50)),'p95':float(np.percentile(lat,95))},
                  feature_mse=float((r-z).square().mean()),policy_hash=policy['hash'])
    if reference_system is not None:
        result['latency_ms']=result.pop('latency_replay_ms')
    path.write_text(json.dumps([{'label':v.tolist(),'top1':a,'top5':b,'digit_correct':c.tolist()} for v,a,b,c in zip(y,top1,top5,(torch.stack(probs).argmax(-1)==y).int())]))
    return result


def fit_linear_probe(train_x,train_y,test_x,test_y):
    mean=train_x.mean(0);std=train_x.std(0).clamp_min(.01)
    a=(train_x-mean)/std;b=(test_x-mean)/std
    torch.manual_seed(901);head=torch.nn.Linear(128,10);opt=torch.optim.Adam(head.parameters(),lr=.01)
    for _ in range(600):
        opt.zero_grad(set_to_none=True);loss=F.cross_entropy(head(a),train_y);loss.backward();opt.step()
    with torch.no_grad():return float((head(b).argmax(-1)==test_y).float().mean())


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    (ROOT/'config.json').write_text(json.dumps(C,indent=2));data=load_data()
    clean_val=clean_waveforms(data['val'][1],619);clean_test=ap.scenarios(data)['clean'][0]
    val_x=torch.cat([data['val'][0],clean_val]);val_y=data['val'][1].repeat(2,1)
    scenarios={'corrupted':(data['test'][0],data['test'][1]),'clean':(clean_test,data['test'][1])}
    result={'config':C,'seeds':{}}
    for seed in C['seeds']:
        folder=ROOT/f'seed_{seed}';folder.mkdir(exist_ok=True)
        cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.encoder.frontend=RelativeFloorLogMel(CFG,.001);cnn.eval()
        for p in cnn.parameters():p.requires_grad_(False)
        tok,meta,system=reference(seed,cnn)
        with torch.no_grad():
            ztrain=tok.project(ap.features(cnn,data['train'][0]));zval=tok.project(ap.features(cnn,val_x))
            target=tok(ztrain)['codes']
        policy={s:freeze_policy(system,x,y,folder/f'{s}_branches.json') for s,(x,y) in scenarios.items()}
        # Replay policy overhead without frontend/projection; these are timed
        # in each candidate evaluation and must not be double-counted.
        policy_times={}
        with torch.inference_mode():
            from ppf.safe_gate import gate_features
            for s,(x,y) in scenarios.items():
                z=tok.project(ap.features(cnn,x)).reshape(len(x),4,4,32);times=[]
                for i,events in enumerate(x):
                    start=time.perf_counter();base=policy[s]['base'][i];zz=z[i];rms=events.square().mean(-1).sqrt()
                    proxy=(system.verifier(zz)/meta['temperatures']['local']).softmax(-1)
                    route=system.gate.score(gate_features(base,proxy,rms,zz))>=meta['route_threshold']
                    _=route&(system.gate.score(gate_features(base,proxy,rms,zz))>=meta['accept_threshold'])
                    times.append((time.perf_counter()-start)*1000)
                policy_times[s]=times
        torch.manual_seed(seed);flat=EMACodebook(64,32)
        loader=DataLoader(TensorDataset(ztrain),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed))
        for _ in range(C['flat_tokenizer_epochs']):
            flat.train()
            for (x,) in loader:flat(x)
        flat.eval();torch.save(flat.state_dict(),folder/'flat_codebook.pt')
        torch.manual_seed(seed);head_state=copy.deepcopy(LocalReranker().state_dict())
        variants={}
        flops={}
        for kind in ['A','B','C','D']:
            torch.manual_seed(seed);m=HierarchyStudyVariant(kind,copy.deepcopy(tok),copy.deepcopy(flat),head_state)
            flops[kind]=training_flops(m,ztrain[:128],data['train'][1].flatten()[:128],[v[:128] for v in target]);variants[kind]=m
        budget=flops['D']*C['equal_update_epochs']*len(ztrain)/128
        output={'profiled_training_flops_per_128':flops,'target_training_flops':budget,'policy_hashes':{s:v['hash'] for s,v in policy.items()},'variants':{}}
        # Temperatures use the unchanged gate's fitting half of validation.
        fit_ids=torch.tensor(meta['gate_validation']['fit_recordings']);calib_ids=torch.cat([fit_ids,fit_ids+len(data['val'][1])])
        for kind,model in variants.items():
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=C['learning_rate'])
            loader=DataLoader(TensorDataset(ztrain,data['train'][1].flatten(),*target),batch_size=C['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
            epoch=0;used=0;steps=0;begin=time.perf_counter();snapshots={}
            while True:
                model.train()
                for batch in loader:
                    x,y,*codes=batch;opt.zero_grad(set_to_none=True);loss=model.loss(x,y,codes);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                    used+=flops[kind]*len(x)/128;steps+=1
                    if epoch>=C['equal_update_epochs'] and used>=budget:break
                epoch+=1
                if epoch==C['equal_update_epochs'] or (epoch>C['equal_update_epochs'] and used>=budget):
                    mode='equal_updates' if epoch==C['equal_update_epochs'] else 'matched_flops'
                    model.eval()
                    with torch.no_grad():temp=ap.temperature(model(zval).reshape(-1,4,10)[calib_ids].flatten(0,1),val_y[calib_ids].flatten())
                    row={'epochs_touched':epoch,'steps':steps,'estimated_training_flops':used,'temperature':temp,'training_seconds_so_far':time.perf_counter()-begin,'scenarios':{}}
                    torch.save(model.state_dict(),folder/f'{kind}_{mode}.pt')
                    for s,(x,y) in scenarios.items():
                        scores=evaluate(model,cnn,tok,x,y,policy[s],temp,folder/f'{kind}_{mode}_{s}_trace.json',system)
                        # Add mean policy-only runtime as a transparent common
                        # overhead approximation, not exact percentile addition.
                        scores['common_policy_overhead_mean_ms']=float(np.mean(policy_times[s]))
                        row['scenarios'][s]=scores
                    with torch.no_grad():
                        train_r=model.representation(ztrain).flatten(1).clone()
                        test_z=tok.project(ap.features(cnn,data['test'][0]));test_r=model.representation(test_z).flatten(1).clone()
                    row['linear_probe_digit_accuracy']=fit_linear_probe(train_r,data['train'][1].flatten(),test_r,data['test'][1].flatten())
                    snapshots[mode]=row
                    print(json.dumps({'seed':seed,'variant':kind,'budget':mode,'corrupted':row['scenarios']['corrupted']}),flush=True)
                    if kind=='D' or used>=budget:
                        if 'matched_flops' not in snapshots:snapshots['matched_flops']=copy.deepcopy(row)
                        break
                if 'matched_flops' in snapshots:break
            output['variants'][kind]=snapshots
        result['seeds'][str(seed)]=output;(ROOT/'metrics.json').write_text(json.dumps(result,indent=2))
    paths=['ppf/hierarchy_study.py','experiments/hierarchy_contribution.py','experiments/hierarchy_config.json','representation_results/cnn.pt','gate_results/metrics.json']
    (ROOT/'source_hashes.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},indent=2))

if __name__=='__main__':main()
