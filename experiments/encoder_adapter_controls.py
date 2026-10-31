"""Probe adapter losses and apply continuous probes under the exact NCM policy."""
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
from ppf.safe_gate import RelativeFloorLogMel
from ppf.encoder_study import ChannelProjection,DigitProbe,digit_input,encoder_taps
from encoder_information import ROOT,C,extract,missing_frequency,digit_metrics,frequency_probe,protected_ncm


class ProbeMemory(torch.nn.Module):
    def __init__(self,probe):super().__init__();self.probe=probe
    def forward(self,z):return {'digit_logits':[self.probe(F.pad(z,(0,32)).flatten(1))]}


def main():
    torch.set_num_threads(1);d=json.loads((ROOT/'metrics.json').read_text());data=load_data()
    tx=torch.cat([data['train'][0],clean_waveforms(data['train'][1],817)]);ty=data['train'][1].repeat(2,1)
    vx=torch.cat([data['val'][0],clean_waveforms(data['val'][1],619)]);vy=data['val'][1].repeat(2,1)
    noise=torch.tensor([v=='noise' for v in data['test'][2]]);partial=torch.tensor([v=='partial_erasure' for v in data['test'][2]])
    scenarios={'corrupted':(data['test'][0],data['test'][1]),'clean':(ap.scenarios(data)['clean'][0],data['test'][1]),
        'noisy':(data['test'][0][noise],data['test'][1][noise]),'partial_erasure':(data['test'][0][partial],data['test'][1][partial]),
        'missing_frequency':(missing_frequency(data['test'][1],818),data['test'][1])}
    for seed,saved in d['seeds'].items():
        folder=ROOT/f'seed_{seed}';cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.encoder.frontend=RelativeFloorLogMel(CFG,.001);cnn.eval()
        tok,meta,system=reference(int(seed),cnn);states=torch.load(folder/'adapters.pt',weights_only=True);adapters={}
        for k,state in states.items():
            adapter=ChannelProjection(len(state['mean']));adapter.load_state_dict(state);adapters[k]=adapter
        train_taps=extract(cnn,tx);val_taps=extract(cnn,vx);taps={s:extract(cnn,x) for s,(x,y) in scenarios.items()}
        fit=torch.tensor(meta['gate_validation']['fit_recordings']);ids=torch.cat([fit,fit+len(data['val'][1])])
        for stage in ['stem_pca','block2_pca']:
            xin=digit_input(train_taps,stage,adapters).detach();vin=digit_input(val_taps,stage,adapters).detach()
            torch.manual_seed(int(seed));probe=DigitProbe(xin.mean(0),xin.std(0).clamp_min(.01));opt=torch.optim.AdamW(probe.parameters(),lr=C['learning_rate'])
            loader=DataLoader(TensorDataset(xin,ty.flatten()),batch_size=C['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(int(seed)))
            for _ in range(C['epochs']):
                for a,b in loader:opt.zero_grad(set_to_none=True);loss=F.cross_entropy(probe(a),b);loss.backward();opt.step()
            probe.eval()
            with torch.no_grad():temp=ap.temperature(probe(vin).reshape(-1,4,10)[ids].flatten(0,1),vy[ids].flatten())
            row={'parameters':sum(p.numel() for p in probe.parameters()),'temperature':temp,'scenarios':{}}
            for s,(x,y) in scenarios.items():
                tin=digit_input(taps[s],stage,adapters).detach()
                with torch.no_grad():row['scenarios'][s]=digit_metrics(probe(tin),y,temp)
                row['scenarios'][s]['frequency_recovery']=frequency_probe(xin,frequency_evidence(tx.flatten(0,1)).mean(1),tin,frequency_evidence(x.flatten(0,1)).mean(1))
                times=[]
                with torch.inference_mode():
                    for a in x:
                        start=time.perf_counter();probe(digit_input(encoder_taps(cnn,a,stage[:-4]),stage,adapters));times.append((time.perf_counter()-start)*1000)
                row['scenarios'][s]['latency_ms']={'p50':float(np.percentile(times,50)),'p95':float(np.percentile(times,95))}
            torch.save(probe.state_dict(),folder/f'{stage}_digit_probe.pt');saved['digit_probes'][stage]=row
            saved['code_probes'][stage]=copy.deepcopy(saved['code_probes'][stage[:-4]])
        policies={}
        for s in scenarios:
            text=(folder/f'{s}_branches.json').read_text();rows=json.loads(text)
            policies[s]={'route':torch.tensor([[len(c)==10 for c in row['choices']] for row in rows]),'accept':torch.tensor([row['accepted_positions'] for row in rows]),'hash':hashlib.sha256(text.encode()).hexdigest()}
        saved['protected_continuous_probes']={}
        for stage,mode in [('final_pca','final'),('stem_pca','stem'),('block2_pca','block2'),('multiscale','multiscale')]:
            probe=DigitProbe(torch.zeros(256),torch.ones(256));probe.load_state_dict(torch.load(folder/f'{stage}_digit_probe.pt',weights_only=True));probe.eval()
            wrapper=ProbeMemory(probe);temp=saved['digit_probes'][stage]['temperature'];row={}
            for s,(x,y) in scenarios.items():row[s]=protected_ncm(wrapper,mode,adapters,cnn,system,tok,x,y,policies[s],temp,folder/f'probe_{stage}_{s}_trace.json')
            saved['protected_continuous_probes'][stage]=row
        (ROOT/'metrics.json').write_text(json.dumps(d,indent=2));print(seed,'adapter and controlled probe comparisons saved',flush=True)

if __name__=='__main__':
    from ppf.graph_ncm import frequency_evidence
    main()
