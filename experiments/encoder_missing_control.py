"""Readout-adaptation control: decode surviving row evidence, not missing digits."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from ppf.model import CNNBaseline
from ppf.safe_gate import RelativeFloorLogMel
from ppf.encoder_study import ChannelProjection,DigitProbe,digit_input,missing_frequency
from compositional_probes import CFG,load_data
from encoder_information import ROOT,C,extract


def compatible(labels):
    rows=torch.where(labels==0,3,(labels-1)//3)
    classes=torch.where(torch.arange(10)==0,3,(torch.arange(10)-1)//3)
    mask=rows.flatten()[:,None]==classes[None]
    return rows.flatten(),mask.float()/mask.sum(-1,keepdim=True)


def main():
    torch.set_num_threads(1);d=json.loads((ROOT/'metrics.json').read_text());data=load_data()
    train=missing_frequency(data['train'][1],819);test=missing_frequency(data['test'][1],818)
    train_rows,targets=compatible(data['train'][1]);test_rows,ideal=compatible(data['test'][1])
    classes=torch.where(torch.arange(10)==0,3,(torch.arange(10)-1)//3)
    for seed,saved in d['seeds'].items():
        folder=ROOT/f'seed_{seed}';cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.encoder.frontend=RelativeFloorLogMel(CFG,.001);cnn.eval()
        taps=extract(cnn,train);testing=extract(cnn,test);states=torch.load(folder/'adapters.pt',weights_only=True);adapters={}
        for k,state in states.items():
            a=ChannelProjection(len(state['mean']));a.load_state_dict(state);adapters[k]=a
        result={}
        for stage in saved['digit_probes']:
            xin=digit_input(taps,stage,adapters).detach();tin=digit_input(testing,stage,adapters).detach()
            torch.manual_seed(int(seed));probe=DigitProbe(xin.mean(0),xin.std(0).clamp_min(.01));opt=torch.optim.AdamW(probe.parameters(),lr=C['learning_rate'])
            loader=DataLoader(TensorDataset(xin,targets),batch_size=C['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(int(seed)))
            for _ in range(C['epochs']):
                for x,y in loader:
                    opt.zero_grad(set_to_none=True);loss=-(y*probe(x).log_softmax(-1)).sum(-1).mean();loss.backward();opt.step()
            probe.eval()
            with torch.inference_mode():
                p=probe(tin).softmax(-1);row_probs=torch.stack([p[:,classes==r].sum(-1) for r in range(4)],-1)
                mass=(p*(ideal>0)).sum(-1)
                result[stage]={'surviving_row_accuracy':float((row_probs.argmax(-1)==test_rows).float().mean()),
                    'compatible_digit_probability_mass':float(mass.mean()),
                    'kl_from_uniform_compatible_posterior':float((ideal*(ideal.clamp_min(1e-30).log()-p.clamp_min(1e-30).log())).sum(-1).mean()),
                    'mean_max_probability_three_digit_rows':float(p[test_rows<3].max(-1).values.mean())}
            torch.save(probe.state_dict(),folder/f'{stage}_missing_aware_probe.pt')
        saved['missing_aware_probes']=result;(ROOT/'metrics.json').write_text(json.dumps(d,indent=2));print(seed,result,flush=True)

if __name__=='__main__':main()
