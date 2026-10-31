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
from encoder_information import extract,digit_metrics,missing_frequency
from ppf.model import CNNBaseline
from ppf.safe_gate import RelativeFloorLogMel,gate_features
from ppf.appt import possibility_tree
from ppf.encoder_study import ChannelProjection,ConditionedNCM
from ppf.alignment_study import LosslessEvidenceAdapter,coarse_summary,padded,CoarseCodeProbe,level1_path,oracle_rollout,aligned_loss

ROOT=Path('alignment_results');C=json.loads(Path('experiments/alignment_config.json').read_text())


def current_adapter(seed):
    state=torch.load(Path('encoder_results')/f'seed_{seed}'/'adapters.pt',weights_only=True)['final']
    adapter=ChannelProjection(64);adapter.load_state_dict(state);return adapter

@torch.inference_mode()
def transform(cnn,x,adapter,lossless):
    taps=extract(cnn,x);anchors=adapter(taps['final'])
    return {'stem':taps['stem'],'final':taps['final'],'original':anchors,
            'lossless':lossless(taps['stem'],taps['final'],anchors)}

@torch.inference_mode()
def raw_metrics(model,memory,y,target,temp):
    digit=[];codes=[[],[],[]];oracle_digit=[];oracle_codes=[[],[],[]]
    for offset in range(0,len(memory),128):
        z=memory[offset:offset+128];out=model(z);oracle=oracle_rollout(model,z,target[0][offset:offset+128])
        digit.append(out['digit_logits'][-1]);oracle_digit.append(oracle['digit_logits'])
        for l in range(3):codes[l].append(out['codes'][l]);oracle_codes[l].append(oracle['codes'][l])
    result=digit_metrics(torch.cat(digit),y,temp)
    result['code_accuracy']=[float((torch.cat(a)==b).float().mean()) for a,b in zip(codes,target)]
    result['oracle_level1']={**digit_metrics(torch.cat(oracle_digit),y,temp),
        'code_accuracy':[float((torch.cat(a)==b).float().mean()) for a,b in zip(oracle_codes,target)]}
    return result

@torch.inference_mode()
def protected(model,kind,cnn,tok,system,adapter,lossless,x,y,policy,temp,path):
    p_all=[];top1=[];top5=[];lat=[];good=bad=0
    for _ in range(3):
        taps=extract(cnn,x[:1]);anchors=adapter(taps['final']);mem=anchors if kind=='original' else lossless(taps['stem'],taps['final'],anchors);model(mem)
    for i,events in enumerate(x):
        start=time.perf_counter();taps=extract(cnn,events[None]);local=tok.project(taps['final'])
        base=(cnn.head(taps['final'])/system.temperatures['cnn']).softmax(-1)
        proxy=(system.verifier(local)/system.temperatures['local']).softmax(-1);rms=events.square().mean(-1).sqrt()
        route=system.gate.score(gate_features(base,proxy,rms,local))>=system.route_cutoff
        accept=route&(system.gate.score(gate_features(base,proxy,rms,local))>=system.accept_cutoff)
        assert torch.equal(route,policy['route'][i]) and torch.equal(accept,policy['accept'][i])
        p=base.clone();chosen=policy['accept'][i]
        if chosen.any():
            anchor=adapter(taps['final'][chosen]);memory=anchor if kind=='original' else lossless(taps['stem'][chosen],taps['final'][chosen],anchor)
            p[chosen]=(model(memory)['digit_logits'][-1]/temp).softmax(-1)
        tree=possibility_tree(p,policy['route'][i],5);lat.append((time.perf_counter()-start)*1000)
        before=base.argmax(-1)==y[i];after=p.argmax(-1)==y[i];good+=int(((~before)&after).sum());bad+=int((before&(~after)).sum())
        p_all.append(p);top1.append(bool((tree['sequences'][0]==y[i]).all()));top5.append(bool((tree['sequences']==y[i]).all(-1).any()))
    result=ap.calibration(torch.cat(p_all),y.flatten());result.update(sequence_top1=float(np.mean(top1)),sequence_top5=float(np.mean(top5)),
       wrong_to_correct=good,correct_to_wrong=bad,latency_ms={'p50':float(np.percentile(lat,50)),'p95':float(np.percentile(lat,95))},policy_hash=policy['hash'])
    path.write_text(json.dumps([{'label':v.tolist(),'top1':a,'top5':b,'digit_correct':c.tolist()} for v,a,b,c in zip(y,top1,top5,(torch.stack(p_all).argmax(-1)==y).int())]))
    return result


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);(ROOT/'config.json').write_text(json.dumps(C,indent=2))
    data=load_data();tx=torch.cat([data['train'][0],clean_waveforms(data['train'][1],817)]);ty=data['train'][1].repeat(2,1)
    vx=torch.cat([data['val'][0],clean_waveforms(data['val'][1],619)]);vy=data['val'][1].repeat(2,1)
    noise=torch.tensor([v=='noise' for v in data['test'][2]]);partial=torch.tensor([v=='partial_erasure' for v in data['test'][2]])
    scenarios={'corrupted':(data['test'][0],data['test'][1]),'clean':(ap.scenarios(data)['clean'][0],data['test'][1]),
        'noisy':(data['test'][0][noise],data['test'][1][noise]),'partial_erasure':(data['test'][0][partial],data['test'][1][partial]),
        'missing_frequency':(missing_frequency(data['test'][1],818),data['test'][1])}
    result={'config':C,'seeds':{}}
    for seed in C['seeds']:
        folder=ROOT/f'seed_{seed}';folder.mkdir(exist_ok=True);cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.encoder.frontend=RelativeFloorLogMel(CFG,.001);cnn.eval()
        for p in cnn.parameters():p.requires_grad_(False)
        tok,meta,system=reference(seed,cnn);adapter=current_adapter(seed);lossless=LosslessEvidenceAdapter()
        training=extract(cnn,tx);lossless.fit(training['stem'][:4800],training['final'][:4800]);torch.save(lossless.state_dict(),folder/'lossless_adapter.pt')
        with torch.no_grad():
            anchors=adapter(training['final']);train={'stem':training['stem'],'final':training['final'],'original':anchors,'lossless':lossless(training['stem'],training['final'],anchors)}
            target=tok(anchors)['codes'];del training
        val=transform(cnn,vx,adapter,lossless);testing={s:transform(cnn,x,adapter,lossless) for s,(x,y) in scenarios.items()}
        test_target={s:tok(v['original'])['codes'] for s,v in testing.items()}
        baseline=ConditionedNCM(copy.deepcopy(tok));baseline.load_state_dict(torch.load(Path('encoder_results')/f'seed_{seed}/final_ncm.pt',weights_only=True));baseline.eval()
        fit=torch.tensor(meta['gate_validation']['fit_recordings']);ids=torch.cat([fit,fit+len(data['val'][1])])
        policies={}
        for s in scenarios:
            path=Path('encoder_results')/f'seed_{seed}/{s}_branches.json';text=path.read_text();rows=json.loads(text);(folder/f'{s}_branches.json').write_text(text)
            policies[s]={'route':torch.tensor([[len(v)==10 for v in row['choices']] for row in rows]),'accept':torch.tensor([row['accepted_positions'] for row in rows]),'hash':hashlib.sha256(text.encode()).hexdigest()}
        # Probe the trained, actually used conditioning path before changing it.
        probe_train={};probe_test={s:{} for s in scenarios};path_train=[];path_test={s:[] for s in scenarios}
        with torch.inference_mode():
            for z in train['original'].split(128):path_train.append({k:v.clone() for k,v in level1_path(baseline,z).items()})
            for s,v in testing.items():
                for z in v['original'].split(128):path_test[s].append({k:a.clone() for k,a in level1_path(baseline,z).items()})
        for name in ['native_final','native_stem_final','current_adapter','lossless_adapter','actual_memory','transformer_hidden']:
            def values(v,path,name=name):
                if name=='native_final':return padded(F.adaptive_avg_pool1d(v['final'],4))
                if name=='native_stem_final':return coarse_summary(v['stem'],v['final'])
                if name=='current_adapter':return padded(v['original'])
                if name=='lossless_adapter':
                    a,b=lossless.unpack(v['lossless']);return coarse_summary(a,b)
                key='projected' if name=='actual_memory' else 'hidden'
                return padded(torch.cat([p[key] for p in path]))
            probe_train[name]=values(train,path_train).detach().clone()
            for s,v in testing.items():probe_test[s][name]=values(v,path_test[s]).detach().clone()
        probes={}
        for name,x in probe_train.items():
            torch.manual_seed(seed);probe=CoarseCodeProbe(x.mean(0),x.std(0).clamp_min(.01));opt=torch.optim.AdamW(probe.parameters(),lr=C['learning_rate'])
            loader=DataLoader(TensorDataset(x,target[0].flatten()),batch_size=C['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
            for _ in range(C['epochs']):
                for a,b in loader:opt.zero_grad(set_to_none=True);loss=F.cross_entropy(probe(a),b);loss.backward();opt.step()
            probe.eval();row={}
            with torch.inference_mode():
                for s,z in probe_test.items():
                    logits=probe(z[name]);labels=test_target[s][0].flatten();row[s]={'coarse_code_accuracy':float((logits.argmax(-1)==labels).float().mean()),'code_nll':float(F.cross_entropy(logits,labels))}
            torch.save(probe.state_dict(),folder/f'{name}_coarse_probe.pt');probes[name]=row;print(seed,'coarse probe',name,row['corrupted'],flush=True)
        with torch.inference_mode():
            e=tok.books[0].weight.double();z=testing['corrupted']['original'].double().mean(1)
            analytic=(2*z@e.t()-e.square().sum(-1)).argmax(-1);truth=test_target['corrupted'][0].flatten()
            W=baseline.predictor.audio_proj.weight.double();b=baseline.predictor.audio_proj.bias.double();pos=baseline.audio_position.double().mean(1)
            projected=torch.cat([p['projected'] for p in path_test['corrupted']]).double().mean(1)
            recovered=(projected-b)@torch.linalg.pinv(W).t()-pos
            analytic_memory=(2*recovered@e.t()-e.square().sum(-1)).argmax(-1)
            a,b=lossless.unpack(testing['corrupted']['lossless'])
            audit={'analytic_adapter_coarse_accuracy':float((analytic==truth).float().mean()),'analytic_after_audio_projection_accuracy':float((analytic_memory==truth).float().mean()),
              'audio_projection_singular_values':torch.linalg.svdvals(W).tolist(),
              'lossless_reconstruction_max_error':max(float((a-testing['corrupted']['stem']).abs().max()),float((b-testing['corrupted']['final']).abs().max())),
              'anchor_max_error':float((testing['corrupted']['lossless'][:,:4]-testing['corrupted']['original']).abs().max()),
              'path_logits_max_difference':float((torch.cat([p['logits'] for p in path_test['corrupted']])-baseline(testing['corrupted']['original'])['code_logits'][0]).abs().max()),
              'target_indices_min_max':[int(target[0].min()),int(target[0].max())],'attention_masks':'None; all memory/code positions included','loss_mask':'all 1/2/4 code positions supervised; no ignored labels'}
        output={'probes':probes,'audit':audit,'models':{}}
        for variant in C['main_variants']+['coarse_only']:
            kind='lossless' if variant.startswith('lossless') else 'original';torch.manual_seed(seed);model=ConditionedNCM(copy.deepcopy(tok))
            if variant=='original_joint':
                model.load_state_dict(baseline.state_dict());training_seconds=0.;history=[]
            else:
                opt=torch.optim.AdamW(model.parameters(),lr=C['learning_rate']);memory=train[kind]
                loader=DataLoader(TensorDataset(memory,ty.flatten(),*target),batch_size=C['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed));history=[];start=time.perf_counter()
                for epoch in range(C['epochs']):
                    model.train();losses=[]
                    for z,y,*codes in loader:
                        opt.zero_grad(set_to_none=True)
                        if variant=='coarse_only':loss=F.cross_entropy(level1_path(model,z)['logits'].flatten(0,1),codes[0].flatten())
                        elif variant.endswith('aligned'):loss=aligned_loss(model,z,y,codes)
                        else:loss=model.loss(z,y,codes)
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
                    history.append(float(np.mean(losses)));print(seed,variant,epoch+1,history[-1],flush=True)
                training_seconds=time.perf_counter()-start
            model.eval();torch.save(model.state_dict(),folder/f'{variant}.pt')
            row={'history':history,'training_seconds':training_seconds,'parameters':sum(p.numel() for p in model.parameters()),'memory_tokens':train[kind].shape[1],'scenarios':{}}
            if variant=='coarse_only':
                with torch.inference_mode():
                    for s,v in testing.items():
                        logits=torch.cat([level1_path(model,z)['logits'] for z in v[kind].split(128)]).flatten(0,1)
                        row['scenarios'][s]={'coarse_code_accuracy':float((logits.argmax(-1)==test_target[s][0].flatten()).float().mean())}
            else:
                with torch.no_grad():
                    logits=model(val[kind])['digit_logits'][-1].reshape(-1,4,10)[ids].flatten(0,1);temp=ap.temperature(logits,vy[ids].flatten())
                row['temperature']=temp
                for s,(x,y) in scenarios.items():row['scenarios'][s]={'raw':raw_metrics(model,testing[s][kind],y,test_target[s],temp),
                    'protected':protected(model,kind,cnn,tok,system,adapter,lossless,x,y,policies[s],temp,folder/f'{variant}_{s}_trace.json')}
            output['models'][variant]=row;result['seeds'][str(seed)]=output;(ROOT/'metrics.json').write_text(json.dumps(result,indent=2));print(seed,'saved',variant,flush=True)

if __name__=='__main__':main()
