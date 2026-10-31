import copy,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import appt_experiment as ap
from compositional_probes import CFG,load_data
from clean_failure_audit import clean_waveforms
from hierarchy_contribution import reference,freeze_policy
from ppf.model import CNNBaseline
from ppf.appt import possibility_tree,LocalReranker
from ppf.sequence import mixture_scores
from ppf.safe_gate import RelativeFloorLogMel,gate_features
from ppf.graph_ncm import frequency_evidence
from ppf.encoder_study import encoder_taps,ChannelProjection,digit_input,condition_input,stage_features,DigitProbe,ConditionedNCM,missing_frequency

ROOT=Path('encoder_results');C=json.loads(Path('experiments/encoder_config.json').read_text())

@torch.inference_mode()
def extract(cnn,x):
    pieces={k:[] for k in ['mel','stem','block1','block2','final']}
    for batch in x.flatten(0,1).split(128):
        for k,v in encoder_taps(cnn,batch).items():pieces[k].append(v)
    return {k:torch.cat(v) for k,v in pieces.items()}


@torch.inference_mode()
def digit_metrics(logits,labels,temp=1.):
    p=(logits/temp).softmax(-1);m=ap.calibration(p,labels.flatten())
    probabilities=p.reshape(-1,1,4,10);scores=mixture_scores(probabilities);top=scores.topk(5,-1).indices
    ids=labels@torch.tensor([1000,100,10,1])
    m.update(sequence_top1=float((top[:,0]==ids).float().mean()),sequence_top5=float((top==ids[:,None]).any(-1).float().mean()),
             digit_accuracy_by_position=(p.argmax(-1).reshape(-1,4)==labels).float().mean(0).tolist(),
             mean_max_probability=float(p.max(-1).values.mean()))
    return m


def frequency_probe(train_x,train_band,test_x,test_band):
    mean=train_x.mean(0);std=train_x.std(0).clamp_min(.01);a=(train_x-mean)/std;b=(test_x-mean)/std
    ym=train_band.mean(0);ys=train_band.std(0).clamp_min(.01);y=(train_band-ym)/ys
    a=torch.cat([a,torch.ones(len(a),1)],1);b=torch.cat([b,torch.ones(len(b),1)],1)
    weight=torch.linalg.solve(a.t()@a+.1*torch.eye(a.shape[1]),a.t()@y)
    error=((b@weight-(test_band-ym)/ys).square()).mean(0)
    return {'train_standardized_band_mse':float(error.mean()),'per_frequency_mse':error.tolist()}


def audit(cnn,events):
    names={};handles=[]
    for name,module in cnn.encoder.named_modules():
        if isinstance(module,(torch.nn.Conv1d,torch.nn.GroupNorm,torch.nn.GELU)):
            def hook(m,args,out,n=name):names[n]={'input':list(args[0].shape),'output':list(out.shape),'operation':repr(m)}
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():taps=encoder_taps(cnn,events);reference_output=cnn.encoder(events)
    for h in handles:h.remove()
    frontend=cnn.encoder.frontend;spec=torch.stft(events,256,80,window=frontend.window,return_complex=True)
    mel=torch.einsum('mf,bft->bmt',frontend.mel,spec.abs().square())
    bins=[]
    for i,row in enumerate(frontend.mel):
        active=(row>0).nonzero().flatten();bins.append({'mel_bin':i,'peak_hz':int(row.argmax())*31.25,
            'nonzero_min_hz':int(active.min())*31.25,'nonzero_max_hz':int(active.max())*31.25})
    return {'waveform':list(events.shape),'stft_complex':list(spec.shape),'stft_power':list(spec.shape),'mel_power':list(mel.shape),
        'taps':{k:list(v.shape) for k,v in taps.items()},'modules':names,'mel_support_hz':bins,
        'manual_taps_equal_native_encoder':bool(torch.equal(taps['final'],reference_output)),
        'gain_halving_max_frontend_difference':float((frontend(events*.5)-frontend(events)).abs().max())}

@torch.inference_mode()
def assess_ncm(model,condition,labels,target,temp):
    codes=[[],[],[]];logits=[]
    for z in condition.split(128):
        out=model(z);logits.append(out['digit_logits'][-1])
        for i,v in enumerate(out['codes']):codes[i].append(v)
    pred=[torch.cat(v) for v in codes]
    result=digit_metrics(torch.cat(logits),labels,temp)
    result['generated_code_accuracy_by_level']=[float((a==b).float().mean()) for a,b in zip(pred,target)]
    return result

@torch.inference_mode()
def protected_ncm(model,mode,adapters,cnn,system,tok,x,y,policy,temp,path):
    lat=[];all_p=[];top1=[];top5=[];helped=harmed=0
    for _ in range(3):model(condition_input(encoder_taps(cnn,x[0]),mode,adapters))
    for i,events in enumerate(x):
        start=time.perf_counter();taps=encoder_taps(cnn,events);local=tok.project(taps['final'])
        base=(cnn.head(taps['final'])/system.temperatures['cnn']).softmax(-1);proxy=(system.verifier(local)/system.temperatures['local']).softmax(-1)
        rms=events.square().mean(-1).sqrt();route=system.gate.score(gate_features(base,proxy,rms,local))>=system.route_cutoff
        accepted=route&(system.gate.score(gate_features(base,proxy,rms,local))>=system.accept_cutoff)
        assert torch.equal(route,policy['route'][i]) and torch.equal(accepted,policy['accept'][i])
        p=base.clone();selected=policy['accept'][i]
        if selected.any():
            condition=condition_input({k:v[selected] for k,v in taps.items()},mode,adapters)
            p[selected]=(model(condition)['digit_logits'][-1]/temp).softmax(-1)
        tree=possibility_tree(p,policy['route'][i],5);lat.append((time.perf_counter()-start)*1000)
        before=base.argmax(-1)==y[i];after=p.argmax(-1)==y[i];helped+=int(((~before)&after).sum());harmed+=int((before&(~after)).sum())
        all_p.append(p);top1.append(bool((tree['sequences'][0]==y[i]).all()));top5.append(bool((tree['sequences']==y[i]).all(-1).any()))
    result=ap.calibration(torch.cat(all_p),y.flatten());result.update(sequence_top1=float(np.mean(top1)),sequence_top5=float(np.mean(top5)),
        wrong_to_correct=helped,correct_to_wrong=harmed,expanded_positions=float(policy['route'].float().sum(-1).mean()),
        candidate_leaves=float((10**policy['route'].sum(-1)).float().mean()),policy_hash=policy['hash'],
        latency_ms={'p50':float(np.percentile(lat,50)),'p95':float(np.percentile(lat,95))})
    path.write_text(json.dumps([{'label':v.tolist(),'top1':a,'top5':b,'digit_correct':c.tolist()} for v,a,b,c in zip(y,top1,top5,(torch.stack(all_p).argmax(-1)==y).int())]))
    return result


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    (ROOT/'config.json').write_text(json.dumps(C,indent=2));data=load_data()
    train_clean=clean_waveforms(data['train'][1],817);val_clean=clean_waveforms(data['val'][1],619);test_clean=ap.scenarios(data)['clean'][0]
    tx=torch.cat([data['train'][0],train_clean]);ty=data['train'][1].repeat(2,1)
    vx=torch.cat([data['val'][0],val_clean]);vy=data['val'][1].repeat(2,1)
    conditions=data['test'][2];all_x=data['test'][0];all_y=data['test'][1]
    noise=torch.tensor([v=='noise' for v in conditions]);partial=torch.tensor([v=='partial_erasure' for v in conditions])
    scenarios={'corrupted':(all_x,all_y),'clean':(test_clean,all_y),'noisy':(all_x[noise],all_y[noise]),
               'partial_erasure':(all_x[partial],all_y[partial]),'missing_frequency':(missing_frequency(all_y,818),all_y)}
    result={'config':C,'seeds':{}}
    for seed in C['seeds']:
        folder=ROOT/f'seed_{seed}';folder.mkdir(exist_ok=True)
        cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.encoder.frontend=RelativeFloorLogMel(CFG,.001);cnn.eval()
        for p in cnn.parameters():p.requires_grad_(False)
        tok,meta,system=reference(seed,cnn)
        if seed==C['seeds'][0]:(ROOT/'encoder_audit.json').write_text(json.dumps(audit(cnn,all_x[0]),indent=2))
        train_taps=extract(cnn,tx);val_taps=extract(cnn,vx);test_taps={s:extract(cnn,x) for s,(x,y) in scenarios.items()}
        adapters={}
        for stage in ['mel','stem','block1','block2','final','multiscale']:
            adapter=ChannelProjection(stage_features(train_taps,stage).shape[1])
            if stage=='final':
                for key in ['mean','std','projection','scale']:getattr(adapter,key).copy_(getattr(tok,key))
            else:adapter.fit(stage_features({k:v[:len(data['train'][1])*4] for k,v in train_taps.items()},stage))
            adapters[stage]=adapter
        torch.save({k:v.state_dict() for k,v in adapters.items()},folder/'adapters.pt')
        with torch.no_grad():
            target=tok(condition_input(train_taps,'final',adapters))['codes']
            test_targets={s:tok(condition_input(v,'final',adapters))['codes'] for s,v in test_taps.items()}
        fit=torch.tensor(meta['gate_validation']['fit_recordings']);calib_ids=torch.cat([fit,fit+len(data['val'][1])])
        output={'digit_probes':{},'code_probes':{},'ncm':{},'direct_quantizer_code_accuracy':{}}
        for s,taps in test_taps.items():
            direct=tok(tok.project(taps['final']))['codes']
            output['direct_quantizer_code_accuracy'][s]=[float((a==b).float().mean()) for a,b in zip(direct,test_targets[s])]
        train_band=frequency_evidence(tx.flatten(0,1)).mean(1)
        for stage in C['probe_stages']:
            torch.manual_seed(seed)
            train_input=digit_input(train_taps,stage,adapters).detach();val_input=digit_input(val_taps,stage,adapters).detach()
            probe=DigitProbe(train_input.mean(0),train_input.std(0).clamp_min(.01))
            opt=torch.optim.AdamW(probe.parameters(),lr=C['learning_rate'])
            loader=DataLoader(TensorDataset(train_input,ty.flatten()),batch_size=C['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
            for _ in range(C['epochs']):
                probe.train()
                for z,y in loader:opt.zero_grad(set_to_none=True);loss=F.cross_entropy(probe(z),y);loss.backward();opt.step()
            probe.eval()
            with torch.no_grad():temp=ap.temperature(probe(val_input).reshape(-1,4,10)[calib_ids].flatten(0,1),vy[calib_ids].flatten())
            row={'parameters':sum(p.numel() for p in probe.parameters()),'temperature':temp,'scenarios':{}}
            for s,(x,y) in scenarios.items():
                test_input=digit_input(test_taps[s],stage,adapters).detach()
                with torch.no_grad():row['scenarios'][s]=digit_metrics(probe(test_input),y,temp)
                row['scenarios'][s]['frequency_recovery']=frequency_probe(train_input,train_band,test_input,frequency_evidence(x.flatten(0,1)).mean(1))
                # Raw probe timings can stop at the requested encoder tap.
                stop=stage if stage in ('mel','stem','block1','block2','final') else None
                times=[]
                with torch.inference_mode():
                    for a in x:
                        start=time.perf_counter();_ = probe(digit_input(encoder_taps(cnn,a,stop),stage,adapters));times.append((time.perf_counter()-start)*1000)
                row['scenarios'][s]['latency_ms']={'p50':float(np.percentile(times,50)),'p95':float(np.percentile(times,95))}
            torch.save(probe.state_dict(),folder/f'{stage}_digit_probe.pt');output['digit_probes'][stage]=row
            # Same 128-input linear canonical-code probe after each 32-channel adapter.
            source=stage[:-4] if stage.endswith('_pca') else 'final' if stage=='final_mean' else stage
            zin=condition_input(train_taps,source,adapters)
            if stage=='final_mean':zin=zin.mean(1,keepdim=True).expand(-1,4,-1)
            xin=zin.flatten(1).detach();mean=xin.mean(0);std=xin.std(0).clamp_min(.01);flat_codes=torch.cat(target,1)
            torch.manual_seed(seed);head=torch.nn.Linear(128,7*64);opt=torch.optim.AdamW(head.parameters(),lr=C['learning_rate'])
            loader=DataLoader(TensorDataset((xin-mean)/std,flat_codes),batch_size=C['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
            for _ in range(C['epochs']):
                for a,b in loader:opt.zero_grad(set_to_none=True);loss=F.cross_entropy(head(a).reshape(-1,7,64).flatten(0,1),b.flatten());loss.backward();opt.step()
            cp={}
            with torch.no_grad():
                for s,taps in test_taps.items():
                    zz=condition_input(taps,source,adapters)
                    if stage=='final_mean':zz=zz.mean(1,keepdim=True).expand(-1,4,-1)
                    hits=head((zz.flatten(1)-mean)/std).reshape(-1,7,64).argmax(-1)==torch.cat(test_targets[s],1)
                    cp[s]=[float(hits[:,:1].float().mean()),float(hits[:,1:3].float().mean()),float(hits[:,3:].float().mean())]
            output['code_probes'][stage]=cp;print(seed,'probe',stage,row['scenarios']['corrupted']['digit_accuracy'],flush=True)
        policies={}
        for s,(x,y) in scenarios.items():policies[s]=freeze_policy(system,x,y,folder/f'{s}_branches.json')
        for mode in C['conditioning']:
            torch.manual_seed(seed);model=ConditionedNCM(copy.deepcopy(tok))
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=C['learning_rate'])
            memory=condition_input(train_taps,mode,adapters).detach()
            loader=DataLoader(TensorDataset(memory,ty.flatten(),*target),batch_size=C['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
            history=[];begin=time.perf_counter()
            for epoch in range(C['epochs']):
                model.train();losses=[]
                for z,y,*codes in loader:
                    opt.zero_grad(set_to_none=True);loss=model.loss(z,y,codes);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
                history.append(float(np.mean(losses)))
            model.eval()
            with torch.no_grad():temp=ap.temperature(model(condition_input(val_taps,mode,adapters))['digit_logits'][-1].reshape(-1,4,10)[calib_ids].flatten(0,1),vy[calib_ids].flatten())
            row={'history':history,'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':time.perf_counter()-begin,'temperature':temp,'scenarios':{}}
            torch.save(model.state_dict(),folder/f'{mode}_ncm.pt')
            for s,(x,y) in scenarios.items():
                row['scenarios'][s]={'raw':assess_ncm(model,condition_input(test_taps[s],mode,adapters),y,test_targets[s],temp),
                    'protected':protected_ncm(model,mode,adapters,cnn,system,tok,x,y,policies[s],temp,folder/f'{mode}_{s}_trace.json')}
            output['ncm'][mode]=row;print(seed,'NCM',mode,row['scenarios']['corrupted'],flush=True)
            result['seeds'][str(seed)]=output;(ROOT/'metrics.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
