"""Additional original-CNN-only controls, with the same fixed compute target."""
import copy,json,hashlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader,TensorDataset
import appt_experiment as ap
from compositional_probes import CFG,load_data
from clean_failure_audit import clean_waveforms
from hierarchy_contribution import reference
from ppf.model import CNNBaseline
from ppf.appt import LocalReranker
from ppf.safe_gate import RelativeFloorLogMel
from ppf.graph_ncm import CNNIdentityProjection,AcousticNCMTransformer
from graph_experiment import ROOT,C,profile,loss_for,logits_for,evaluate


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);d=json.loads((ROOT/'metrics.json').read_text());data=load_data()
    clean=ap.scenarios(data)['clean'][0];vx=torch.cat([data['val'][0],clean_waveforms(data['val'][1],619)]);vy=data['val'][1].repeat(2,1)
    scenarios={'corrupted':(data['test'][0],data['test'][1]),'clean':(clean,data['test'][1])}
    for seed in C['seeds']:
        folder=ROOT/f'seed_{seed}';cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.encoder.frontend=RelativeFloorLogMel(CFG,.001);cnn.eval()
        for p in cnn.parameters():p.requires_grad_(False)
        tok,meta,system=reference(seed,cnn)
        with torch.no_grad():ztrain=tok.project(ap.features(cnn,data['train'][0]));zval=tok.project(ap.features(cnn,vx));targets=tok(ztrain)['codes']
        torch.manual_seed(seed);head=copy.deepcopy(LocalReranker().state_dict())
        saved=d['seeds'][str(seed)];budget=saved['target_training_flops'];policies={}
        for s in scenarios:
            text=(folder/f'{s}_branches.json').read_text();rows=json.loads(text)
            policies[s]={'route':torch.tensor([[len(c)==10 for c in r['choices']] for r in rows]),'accept':torch.tensor([r['accepted_positions'] for r in rows]),'hash':hashlib.sha256(text.encode()).hexdigest()}
        fit=torch.tensor(meta['gate_validation']['fit_recordings']);ids=torch.cat([fit,fit+len(data['val'][1])])
        for kind,ncm in [('A_cnn',False),('C_cnn',True)]:
            torch.manual_seed(seed);model=AcousticNCMTransformer(copy.deepcopy(tok)) if ncm else LocalReranker();(model.ranker if ncm else model).load_state_dict(head)
            f=profile(model,ztrain[:128],data['train'][1].flatten()[:128],[t[:128] for t in targets],ncm);saved['profiled_training_flops_per_128'][kind]=f
            torch.manual_seed(seed);opt=torch.optim.AdamW(model.parameters(),lr=C['learning_rate'])
            loader=DataLoader(TensorDataset(ztrain,data['train'][1].flatten(),*targets),batch_size=C['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
            used=0;steps=0;epoch=0;snapshots={}
            while True:
                model.train()
                for z,y,*target in loader:
                    opt.zero_grad(set_to_none=True);loss=loss_for(model,z,y,target,ncm);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();used+=f*len(z)/128;steps+=1
                    if epoch>=C['epochs'] and used>=budget:break
                epoch+=1
                if epoch==C['epochs'] or (epoch>C['epochs'] and used>=budget):
                    mode='equal_updates' if epoch==C['epochs'] else 'matched_flops';model.eval()
                    with torch.no_grad():temp=ap.temperature(logits_for(model,zval,ncm).reshape(-1,4,10)[ids].flatten(0,1),vy[ids].flatten())
                    row={'epochs_touched':epoch,'steps':steps,'estimated_training_flops':used,'temperature':temp,'parameters':sum(p.numel() for p in model.parameters()),'scenarios':{}}
                    torch.save(model.state_dict(),folder/f'{kind}_{mode}.pt')
                    for s,(x,y) in scenarios.items():row['scenarios'][s]=evaluate(model,ncm,CNNIdentityProjection(),cnn,tok,system,x,y,policies[s],temp,folder/f'{kind}_{mode}_{s}_trace.json')
                    snapshots[mode]=row;print(seed,kind,mode,row['scenarios']['corrupted']['sequence_top1'],flush=True)
                    if used>=budget:
                        if 'matched_flops' not in snapshots:snapshots['matched_flops']=copy.deepcopy(row)
                        break
                if 'matched_flops' in snapshots:break
            saved['variants'][kind]=snapshots
        (ROOT/'metrics.json').write_text(json.dumps(d,indent=2))

if __name__=='__main__':main()
