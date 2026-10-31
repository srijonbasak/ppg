"""Saved-model evaluation with complete common evidence-analysis costs."""
import json
from pathlib import Path
import torch
import appt_experiment as ap
from compositional_probes import CFG,load_data
from hierarchy_contribution import reference
from ppf.model import CNNBaseline
from ppf.safe_gate import RelativeFloorLogMel
from ppf.graph_ncm import load_graph_tree
from graph_experiment import ROOT,evaluate


def main():
    torch.set_num_threads(1);d=json.loads((ROOT/'metrics.json').read_text());data=load_data()
    scenarios={'corrupted':(data['test'][0],data['test'][1]),'clean':(ap.scenarios(data)['clean'][0],data['test'][1])}
    for seed,saved in d['seeds'].items():
        folder=ROOT/f'seed_{seed}';cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.encoder.frontend=RelativeFloorLogMel(CFG,.001);cnn.eval()
        tok,meta,system=reference(int(seed),cnn)
        for kind,budgets in saved['variants'].items():
            for mode,row in budgets.items():
                tree=load_graph_tree(ROOT,int(seed),kind,mode)
                for name,(x,y) in scenarios.items():
                    source=(folder/f'{name}_branches.json');text=source.read_text();choices=json.loads(text)
                    import hashlib
                    policy={'route':torch.tensor([[len(c)==10 for c in v['choices']] for v in choices]),
                            'accept':torch.tensor([v['accepted_positions'] for v in choices]),'hash':hashlib.sha256(text.encode()).hexdigest()}
                    row['scenarios'][name]=evaluate(tree.model,tree.ncm,tree.projection,cnn,tok,system,x,y,policy,row['temperature'],folder/f'{kind}_{mode}_{name}_trace.json')
                print(seed,kind,mode,flush=True)
        (ROOT/'metrics.json').write_text(json.dumps(d,indent=2))

if __name__=='__main__':main()
