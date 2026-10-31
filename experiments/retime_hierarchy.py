"""Evaluate saved models with full common frozen-policy runtime, no retraining."""
import json
from pathlib import Path
import torch
from ppf.model import CNNBaseline
from ppf.appt import LocalReranker,LocalAcousticTokenizer
from ppf.sequence import EMACodebook
from ppf.safe_gate import RelativeFloorLogMel
from ppf.hierarchy_study import HierarchyStudyVariant
from hierarchy_contribution import ROOT,C,reference,evaluate
from compositional_probes import CFG,load_data
import appt_experiment as ap

def main():
    torch.set_num_threads(1);data=load_data();d=json.loads((ROOT/'metrics.json').read_text())
    scenarios={'corrupted':(data['test'][0],data['test'][1]),'clean':(ap.scenarios(data)['clean'][0],data['test'][1])}
    for seed,saved in d['seeds'].items():
        folder=ROOT/f'seed_{seed}';cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.encoder.frontend=RelativeFloorLogMel(CFG,.001);cnn.eval()
        tok,meta,system=reference(int(seed),cnn)
        for kind,budgets in saved['variants'].items():
            for mode,row in budgets.items():
                path=folder/f'{kind}_{mode}.pt'
                if not path.exists():path=folder/f'{kind}_equal_updates.pt'
                model=HierarchyStudyVariant(kind,LocalAcousticTokenizer(),EMACodebook(64,32),LocalReranker().state_dict())
                model.load_state_dict(torch.load(path,weights_only=True));model.eval()
                for name,(x,y) in scenarios.items():
                    choices=json.loads((folder/f'{name}_branches.json').read_text())
                    route=torch.tensor([[len(v)==10 for v in r['choices']] for r in choices])
                    accepted=torch.tensor([r['accepted_positions'] for r in choices])
                    with torch.no_grad():base=(cnn(x.flatten(0,1))/meta['temperatures']['cnn']).softmax(-1).reshape(-1,4,10)
                    policy={'base':base,'route':route,'accept':accepted,'hash':saved['policy_hashes'][name]}
                    row['scenarios'][name]=evaluate(model,cnn,tok,x,y,policy,row['temperature'],folder/f'{kind}_{mode}_{name}_trace.json',system)
                    with torch.inference_mode():
                        z=tok.project(ap.features(cnn,x));row['scenarios'][name]['all_event_candidate_digit_accuracy']=float((model(z).argmax(-1)==y.flatten()).float().mean())
                print(seed,kind,mode,flush=True)
        (ROOT/'metrics.json').write_text(json.dumps(d,indent=2))

if __name__=='__main__':main()
