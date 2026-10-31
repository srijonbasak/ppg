"""Paired fixed-budget continuation; no test-set checkpoint selection.
PYTHONPATH=. .venv/bin/python experiments/debug_prefix_training.py
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from debug_conditional import ROOT,SETTINGS,load_model,diagnostics,wiring_checks

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    data=torch.load('representation_results/audio_data.pt',weights_only=False)
    x,y,_=data['train'];result={'settings':SETTINGS,'scope':'paired single-seed continuation from identical checkpoint; fixed eight extra epochs','runs':{}}
    for name,prob in [('teacher_continuation',0.),('mixed_prefix',.5)]:
        torch.manual_seed(SETTINGS['seed'])
        model=load_model();model.load_state_dict(torch.load('representation_results/ppf.pt',weights_only=True))
        before={k:v.clone() for k,v in model.tokenizer.state_dict().items()}
        # The separate generator fixes example order across both training arms.
        loader=DataLoader(TensorDataset(x,y),batch_size=SETTINGS['batch_size'],shuffle=True,
                          generator=torch.Generator().manual_seed(SETTINGS['seed']))
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=SETTINGS['mixed_learning_rate'])
        history=[];start=time.perf_counter()
        for epoch in range(SETTINGS['mixed_epochs']):
            model.train();losses=[]
            for audio,labels in loader:
                with torch.no_grad():codes=model.tokenizer(labels)['codes']
                opt.zero_grad(set_to_none=True)
                logits=model.training_logits(audio,codes,generated_prefix_probability=prob)
                loss=sum(F.cross_entropy(l.flatten(0,1),c.flatten()) for l,c in zip(logits,codes))/3
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                losses.append(float(loss.detach()))
            history.append(float(np.mean(losses)));print(name,epoch+1,history[-1],flush=True)
        assert all(torch.equal(before[k],v) for k,v in model.tokenizer.state_dict().items())
        torch.save(model.state_dict(),ROOT/f'{name}.pt')
        row={'history':history,'training_seconds':time.perf_counter()-start,'tokenizer_unchanged':True}
        for split in ['val','test']:
            audio,labels,_=data[split]
            row[split]=diagnostics(model,audio,labels)
        row['wiring']=wiring_checks(model,data['val'][0][:32],data['val'][1][:32])
        # Goertzel alignment applies only to original clean tiny set; these
        # mixed-corruption data do not get a clean-audio detector assertion.
        row['wiring'].pop('clean_audio_goertzel_accuracy',None)
        result['runs'][name]=row
        (ROOT/'prefix_comparison.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'run':name,'test':row['test']}),flush=True)

if __name__=='__main__':main()
