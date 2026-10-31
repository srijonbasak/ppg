"""Focused diagnostics; keeps tokenizer, CNN and generator architecture fixed.
PYTHONPATH=. .venv/bin/python experiments/debug_conditional.py
"""
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from ppf.config import ModelConfig
from ppf.data import synthesize_sequence,SynthesisConfig
from ppf.engine import goertzel_predict
from ppf.model import CNNBaseline
from ppf.sequence import CleanLabelTokenizer,SequencePPF,mixture_scores

ROOT=Path('conditional_debug');CFG=ModelConfig()
SETTINGS={'seed':113,'tiny_examples':32,'tiny_max_steps':2000,'tiny_learning_rate':.001,
          'evaluate_every':50,'mixed_prefix_probability':.5,'mixed_epochs':8,
          'mixed_learning_rate':.0003,'batch_size':32,'threads':1,
          'heldout_usage':'fixed checkpoint evaluation only, no selection on test',
          'tiny_stop':'all 32 sequences and all seven codes correct under greedy target-free generation'}

def load_model():
    token=CleanLabelTokenizer(CFG)
    token.load_state_dict(torch.load('representation_results/tokenizer.pt',weights_only=True)['model'])
    model=SequencePPF(token,CFG)
    return model

@torch.inference_mode()
def diagnostics(model,audio,labels):
    model.eval();target=model.tokenizer(labels)['codes']
    tf=model.teacher_logits(audio,target)
    probs,predicted,genlog=model.generate(audio,stochastic=False)
    p=probs[:,0];ranked=mixture_scores(probs).topk(5,-1).indices
    ids=labels@torch.tensor([1000,100,10,1]);digit=p.argmax(-1)==labels
    result={'samples':len(audio),'digit_accuracy':float(digit.float().mean()),
       'sequence_accuracy':float(digit.all(-1).float().mean()),
       'sequence_top5':float((ranked==ids[:,None]).any(-1).float().mean()),
       'oracle_reconstruction':float((model.tokenizer.decode_codes(target).argmax(-1)==labels).all(-1).float().mean()),
       'teacher_code_accuracy':[], 'generated_code_accuracy':[],
       'teacher_code_ce':[], 'generated_prefix_code_ce':[],
       'level1':{}}
    for l in range(3):
        result['teacher_code_accuracy'].append(float((tf[l].argmax(-1)==target[l]).float().mean()))
        result['generated_code_accuracy'].append(float((predicted[l]==target[l]).float().mean()))
        result['teacher_code_ce'].append(float(F.cross_entropy(tf[l].flatten(0,1),target[l].flatten())))
        result['generated_prefix_code_ce'].append(float(F.cross_entropy(genlog[l].flatten(0,1),target[l].flatten())))
    first=tf[0].softmax(-1)
    frequency=torch.bincount(target[0].flatten(),minlength=64).float();frequency/=frequency.sum()
    result['level1']={'target_majority_frequency':float(frequency.max()),
       'mean_max_probability':float(first.max(-1).values.mean()),
       'mean_entropy':float(-(first*first.clamp_min(1e-20).log()).sum(-1).mean()),
       'top5_code_accuracy':float((first.topk(5,-1).indices==target[0][:,:,None]).any(-1).float().mean()),
       'target_frequencies':frequency.tolist()}
    return result


def clean_examples():
    rng=np.random.default_rng(SETTINGS['seed']);audio=[];labels=[];records=[]
    cfg=SynthesisConfig(tone_ms=(180,180),gap_ms=(30,30))
    seen=set()
    while len(audio)<SETTINGS['tiny_examples']:
        y=rng.integers(0,10,4)
        if tuple(y) in seen:continue
        seen.add(tuple(y))
        wav,meta=synthesize_sequence(''.join(map(str,y)),rng,cfg,corruption=False)
        crops=np.stack([np.pad(wav[e['start_sample']:e['end_sample']],(0,960)) for e in meta['events']])
        decoded=[goertzel_predict(c) for c in crops]
        assert decoded==y.tolist()
        audio.append(crops);labels.append(y)
        records.append({'index':len(audio)-1,'labels':y.tolist(),'goertzel':decoded,
                        'audio_sha256':hashlib.sha256(crops.tobytes()).hexdigest()})
    (ROOT/'tiny_manifest.json').write_text(json.dumps(records,indent=2))
    x=torch.tensor(np.stack(audio));y=torch.tensor(np.stack(labels))
    torch.save({'audio':x,'labels':y},ROOT/'tiny_data.pt')
    return x,y

@torch.inference_mode()
def wiring_checks(model,x,y):
    model.eval();token=model.tokenizer(y);target=token['codes']
    expected=model.tokenizer.decode_codes(target).softmax(-1)
    def force_codes(memory,priors,level,length):
        codes=target[level].repeat_interleave(3,0)
        assert length==codes.shape[1]
        for l,q in enumerate(priors):
            assert torch.equal(q,model.tokenizer.books[l].lookup(target[l].repeat_interleave(3,0)))
        logits=torch.full((*codes.shape,64),-100.)
        return logits.scatter_(-1,codes[:,:,None],100.)
    with patch.object(model.predictor,'forward_level',side_effect=force_codes):
        generated,_,_=model.generate(x,samples=3,stochastic=False)
    diff=float((generated-expected[:,None]).abs().max())
    assert diff<1e-6
    # Flattening must preserve event and sequence order exactly.
    raw=model.encoder(x.flatten(0,1)).reshape(len(x),4,64,-1)+model.event_positions
    memory=model.audio_memory(x)
    restored=memory.reshape(len(x),64,4,-1).permute(0,2,1,3)
    assert torch.equal(raw,restored)
    # Teacher path and explicit level calls must be identical in eval mode.
    tf=model.teacher_logits(x,target);previous=[];differences=[]
    for l,c in enumerate(target):
        out=model.predictor.forward_level(memory,previous,l,c.shape[1])
        differences.append(float((out-tf[l]).abs().max()))
        previous.append(model.tokenizer.books[l].lookup(c))
    reversed_first=model.predictor.forward_level(model.audio_memory(x.flip(1)),[],0,1)
    return {'oracle_inference_max_probability_difference':diff,
            'training_vs_manual_level_logits_max_difference':differences,
            'flatten_event_alignment_exact':True,'clean_audio_goertzel_accuracy':1.,
            'reversing_event_order_level1_logit_mean_absolute_change':float((reversed_first-tf[0]).abs().mean()),
            'event_position_rms':float(model.event_positions.square().mean().sqrt()),
            'audio_feature_rms':float(raw.square().mean().sqrt())}


def feature_probe(model,x,y):
    model.eval()
    with torch.no_grad():
        feat=model.encoder(x.flatten(0,1)).mean(-1).reshape(len(x),-1)
        codes=torch.cat(model.tokenizer(y)['codes'],1)
    # Probe only: a linear classifier on frozen, ordered per-event CNN features.
    probe=torch.nn.Linear(feat.shape[-1],7*64)
    opt=torch.optim.Adam(probe.parameters(),lr=.01)
    for _ in range(1000):
        opt.zero_grad(set_to_none=True);out=probe(feat).reshape(-1,7,64)
        loss=F.cross_entropy(out.flatten(0,1),codes.flatten());loss.backward();opt.step()
    with torch.no_grad():
        pred=probe(feat).reshape(-1,7,64).argmax(-1)
        hits=pred==codes
    return {'scope':'training-set linear probe on ordered frozen CNN event means; not generalization',
            'all_code_accuracy':float(hits.float().mean()),
            'all_seven_codes_correct_fraction':float(hits.all(-1).float().mean()),
            'accuracy_by_level':[float(hits[:,:1].float().mean()),float(hits[:,1:3].float().mean()),float(hits[:,3:].float().mean())]}


def overfit(x,y):
    model=load_model()
    cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True))
    model.encoder.load_state_dict(cnn.encoder.state_dict())
    with torch.inference_mode():
        cnn.eval();cnn_acc=float((cnn(x.flatten(0,1)).argmax(-1)==y.flatten()).float().mean())
        targets=model.tokenizer(y)['codes']
    # Clone out of inference mode so loss can save indices for backward.
    targets=[c.clone() for c in targets]
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=SETTINGS['tiny_learning_rate'])
    original={k:v.clone() for k,v in model.tokenizer.state_dict().items()}
    history=[];start=time.perf_counter()
    for step in range(SETTINGS['tiny_max_steps']+1):
        if step%SETTINGS['evaluate_every']==0:
            metrics=diagnostics(model,x,y)
            row={'step':step,'metrics':metrics};history.append(row)
            print(json.dumps({'tiny_step':step,'sequence_accuracy':metrics['sequence_accuracy'],
                              'teacher':metrics['teacher_code_accuracy'],'generated':metrics['generated_code_accuracy']}),flush=True)
            if metrics['sequence_accuracy']>=.999 and min(metrics['generated_code_accuracy'])>=.999:break
        if step==SETTINGS['tiny_max_steps']:break
        model.train();opt.zero_grad(set_to_none=True)
        logits=model.teacher_logits(x,targets)
        loss=sum(F.cross_entropy(l.flatten(0,1),c.flatten()) for l,c in zip(logits,targets))/3
        loss.backward()
        grad={name:float(p.grad.norm()) for name,p in model.named_parameters()
              if p.grad is not None and ('event_positions' in name or 'audio_proj' in name or 'condition_proj' in name or 'level_embedding' in name)}
        total_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));opt.step()
        if step%50==0:
            history[-1]['training_loss']=float(loss.detach());history[-1]['gradient_norm_before_clip']=total_norm
            history[-1]['selected_gradient_norms']=grad
    assert all(torch.equal(original[k],v) for k,v in model.tokenizer.state_dict().items())
    torch.save(model.state_dict(),ROOT/'tiny_model.pt')
    result={'initial_cnn_digit_accuracy':cnn_acc,'training_seconds':time.perf_counter()-start,
            'history':history,'tokenizer_unchanged':True,'final':metrics,
            'linear_feature_probe':feature_probe(model,x,y)}
    (ROOT/'tiny_metrics.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.manual_seed(SETTINGS['seed']);np.random.seed(SETTINGS['seed'])
    torch.use_deterministic_algorithms(True)
    (ROOT/'config.json').write_text(json.dumps(SETTINGS,indent=2))
    x,y=clean_examples()
    original=load_model();original.load_state_dict(torch.load('representation_results/ppf.pt',weights_only=True));original.eval()
    checks=wiring_checks(original,x,y)
    (ROOT/'wiring.json').write_text(json.dumps(checks,indent=2))
    data=torch.load('representation_results/audio_data.pt',weights_only=False)
    results={}
    for split in ['val','test']:
        audio,labels,_=data[split]
        results[split]=diagnostics(original,audio,labels)
    (ROOT/'original_deterministic.json').write_text(json.dumps(results,indent=2))
    # Probe the original frozen features before any tiny-set optimization.
    probe=feature_probe(original,x,y)
    (ROOT/'original_feature_probe.json').write_text(json.dumps(probe,indent=2))
    result=overfit(x,y)
    print(json.dumps({'wiring':checks,'original_feature_probe':probe,'tiny_final':result['final']}),flush=True)

if __name__=='__main__':main()
