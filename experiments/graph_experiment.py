import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import appt_experiment as ap
from compositional_probes import CFG,load_data
from clean_failure_audit import clean_waveforms
from hierarchy_contribution import reference
from ppf.model import CNNBaseline
from ppf.appt import LocalAcousticTokenizer,LocalReranker,possibility_tree
from ppf.safe_gate import RelativeFloorLogMel,gate_features
from ppf.graph_ncm import EvidenceProjection,AcousticNCMTransformer,frequency_evidence,graph_messages,missing_band_distribution

ROOT=Path('graph_results');C=json.loads(Path('experiments/graph_config.json').read_text())
KINDS={'A':('flat',False),'B':('graph',False),'C':('flat',True),'D':('graph',True),'rewired':('rewired',False),'D_no_prior':('graph',True)}


def loss_for(model,z,y,target,ncm):
    return model.loss(z,y,target) if ncm else F.cross_entropy(model(z),y)

@torch.inference_mode()
def logits_for(model,z,ncm):return model(z)['digit_logits'][-1] if ncm else model(z)


def profile(model,z,y,target,ncm):
    model.train();model.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as p:
        loss_for(model,z,y,target,ncm).backward()
    model.zero_grad(set_to_none=True);return sum(v.flops for v in p.key_averages())

@torch.inference_mode()
def evaluate(model,ncm,projection,cnn,tok,system,x,y,policy,temp,path):
    probs=[];top1=[];top5=[];lat=[];recover=harm=0
    for _ in range(3):
        features=cnn.encoder(x[0]);z=projection(tok.project(features),frequency_evidence(x[0]));logits_for(model,z,ncm)
    for i,events in enumerate(x):
        start=time.perf_counter();features=cnn.encoder(events);local=tok.project(features)
        spectral=frequency_evidence(events)
        band_mask,_=missing_band_distribution(events,spectral)
        assert not band_mask.any(), 'Physical guard changed the frozen main-test policy'
        base=(cnn.head(features)/system.temperatures['cnn']).softmax(-1)
        proxy=(system.verifier(local)/system.temperatures['local']).softmax(-1);rms=events.square().mean(-1).sqrt()
        route=system.gate.score(gate_features(base,proxy,rms,local))>=system.route_cutoff
        accepted=route&(system.gate.score(gate_features(base,proxy,rms,local))>=system.accept_cutoff)
        assert torch.equal(route,policy['route'][i]) and torch.equal(accepted,policy['accept'][i])
        p=base.clone();chosen=policy['accept'][i]
        if chosen.any():
            z=projection(local[chosen],spectral[chosen])
            p[chosen]=(logits_for(model,z,ncm)/temp).softmax(-1)
        tree=possibility_tree(p,route,5);lat.append((time.perf_counter()-start)*1000)
        before=base.argmax(-1)==y[i];after=p.argmax(-1)==y[i]
        recover+=int(((~before)&after).sum());harm+=int((before&(~after)).sum())
        probs.append(p);top1.append(bool((tree['sequences'][0]==y[i]).all()));top5.append(bool((tree['sequences']==y[i]).all(-1).any()))
    result=ap.calibration(torch.cat(probs),y.flatten())
    result.update(sequence_top1=float(np.mean(top1)),sequence_top5=float(np.mean(top5)),wrong_to_correct=recover,correct_to_wrong=harm,
          expanded_positions=float(policy['route'].float().sum(-1).mean()),candidate_leaves=float((10**policy['route'].sum(-1)).float().mean()),
          policy_hash=policy['hash'],latency_ms={'p50':float(np.percentile(lat,50)),'p95':float(np.percentile(lat,95))})
    path.write_text(json.dumps([{'label':v.tolist(),'top1':a,'top5':b,'digit_correct':c.tolist()} for v,a,b,c in zip(y,top1,top5,(torch.stack(probs).argmax(-1)==y).int())]))
    return result


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    (ROOT/'config.json').write_text(json.dumps(C,indent=2));data=load_data();clean_test=ap.scenarios(data)['clean'][0]
    clean_val=clean_waveforms(data['val'][1],619)
    vx=torch.cat([data['val'][0],clean_val]);vy=data['val'][1].repeat(2,1)
    scenarios={'corrupted':(data['test'][0],data['test'][1]),'clean':(clean_test,data['test'][1])}
    # Prove that the proposed physical relations have useful signal before
    # fitting graph-conditioned code generators. No model has access to labels here.
    physics={}
    for s,(x,y) in scenarios.items():
        e=frequency_evidence(x.flatten(0,1));true=graph_messages(e).mean(1).argmax(-1)
        rewired=EvidenceProjection(True,True);bad=graph_messages(e,rewired.pairs).mean(1).argmax(-1)
        physics[s]={'physical_pair_score_digit_accuracy':float((true==y.flatten()).float().mean()),'rewired_pair_score_digit_accuracy':float((bad==y.flatten()).float().mean())}
    (ROOT/'physics_check.json').write_text(json.dumps(physics,indent=2))
    results={'config':C,'physics':physics,'seeds':{}}
    for seed in C['seeds']:
        folder=ROOT/f'seed_{seed}';folder.mkdir(exist_ok=True);torch.manual_seed(seed)
        cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.encoder.frontend=RelativeFloorLogMel(CFG,.001);cnn.eval()
        for p in cnn.parameters():p.requires_grad_(False)
        local_tok,meta,system=reference(seed,cnn)
        with torch.no_grad():
            local_train=local_tok.project(ap.features(cnn,data['train'][0]));e_train=frequency_evidence(data['train'][0].flatten(0,1))
            local_val=local_tok.project(ap.features(cnn,vx));e_val=frequency_evidence(vx.flatten(0,1))
        projections={};features={};valfeatures={};tokenizers={};targets={}
        for name in ['flat','graph','rewired']:
            enc=EvidenceProjection(name!='flat',name=='rewired');enc.fit(local_train,e_train);projections[name]=enc
            with torch.no_grad():features[name]=enc(local_train,e_train);valfeatures[name]=enc(local_val,e_val)
            torch.manual_seed(seed);tok=LocalAcousticTokenizer()
            loader=DataLoader(TensorDataset(features[name]),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed))
            for _ in range(C['tokenizer_epochs']):
                tok.train()
                for (z,) in loader:tok(z,cutoff=int(torch.randint(1,4,()).item()))
            tok.freeze();tokenizers[name]=tok
            with torch.no_grad():targets[name]=tok(features[name])['codes']
            torch.save(enc.state_dict(),folder/f'{name}_projection.pt');torch.save(tok.state_dict(),folder/f'{name}_tokenizer.pt')
        policies={}
        for s,(x,y) in scenarios.items():
            source=Path('hierarchy_results')/f'seed_{seed}/{s}_branches.json';rows=json.loads(source.read_text())
            route=torch.tensor([[len(v)==10 for v in row['choices']] for row in rows]);accepted=torch.tensor([row['accepted_positions'] for row in rows])
            text=source.read_text();(folder/f'{s}_branches.json').write_text(text)
            policies[s]={'route':route,'accept':accepted,'hash':hashlib.sha256(text.encode()).hexdigest()}
        torch.manual_seed(seed);head=copy.deepcopy(LocalReranker().state_dict());models={};flops={}
        for name,(feat,ncm) in KINDS.items():
            torch.manual_seed(seed)
            model=AcousticNCMTransformer(copy.deepcopy(tokenizers[feat]),name!='D_no_prior') if ncm else LocalReranker()
            (model.ranker if ncm else model).load_state_dict(head)
            flops[name]=profile(model,features[feat][:128],data['train'][1].flatten()[:128],[v[:128] for v in targets[feat]],ncm);models[name]=model
        budget=max(flops.values())*C['epochs']*len(local_train)/128
        output={'profiled_training_flops_per_128':flops,'target_training_flops':budget,'variants':{}}
        fit_ids=torch.tensor(meta['gate_validation']['fit_recordings']);ids=torch.cat([fit_ids,fit_ids+len(data['val'][1])])
        for name,(feat,ncm) in KINDS.items():
            model=models[name];loader=DataLoader(TensorDataset(features[feat],data['train'][1].flatten(),*targets[feat]),batch_size=C['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
            opt=torch.optim.AdamW(model.parameters(),lr=C['learning_rate']);used=0;epoch=0;steps=0;snapshots={}
            while True:
                model.train()
                for z,y,*target in loader:
                    opt.zero_grad(set_to_none=True);loss=loss_for(model,z,y,target,ncm);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                    used+=flops[name]*len(z)/128;steps+=1
                    if epoch>=C['epochs'] and used>=budget:break
                epoch+=1
                if epoch==C['epochs'] or (epoch>C['epochs'] and used>=budget):
                    mode='equal_updates' if epoch==C['epochs'] else 'matched_flops';model.eval()
                    with torch.no_grad():temp=ap.temperature(logits_for(model,valfeatures[feat],ncm).reshape(-1,4,10)[ids].flatten(0,1),vy[ids].flatten())
                    row={'epochs_touched':epoch,'steps':steps,'estimated_training_flops':used,'temperature':temp,
                         'parameters':sum(p.numel() for p in model.parameters()),'scenarios':{}}
                    torch.save(model.state_dict(),folder/f'{name}_{mode}.pt')
                    for s,(x,y) in scenarios.items():row['scenarios'][s]=evaluate(model,ncm,projections[feat],cnn,local_tok,system,x,y,policies[s],temp,folder/f'{name}_{mode}_{s}_trace.json')
                    snapshots[mode]=row;print(json.dumps({'seed':seed,'variant':name,'budget':mode,'corrupted':row['scenarios']['corrupted']}),flush=True)
                    if used>=budget:
                        if 'matched_flops' not in snapshots:snapshots['matched_flops']=copy.deepcopy(row)
                        break
                if 'matched_flops' in snapshots:break
            output['variants'][name]=snapshots
        results['seeds'][str(seed)]=output;(ROOT/'metrics.json').write_text(json.dumps(results,indent=2))
    paths=['ppf/graph_ncm.py','experiments/graph_config.json','representation_results/cnn.pt','hierarchy_results/metrics.json']
    (ROOT/'source_hashes.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},indent=2))

if __name__=='__main__':main()
