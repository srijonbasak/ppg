"""Distinguish tiny-set memorization from acoustic/label generalization."""
import json
import numpy as np
import torch
from ppf.data import SynthesisConfig,synthesize_sequence
from debug_conditional import ROOT,load_model,diagnostics

def main():
    torch.set_num_threads(1)
    tiny=torch.load(ROOT/'tiny_data.pt',weights_only=True)
    rng=np.random.default_rng(114)
    original=tiny['labels'];seen={tuple(y.tolist()) for y in original}
    unseen=[]
    while len(unseen)<32:
        y=tuple(int(v) for v in rng.integers(0,10,4))
        if y not in seen:unseen.append(y);seen.add(y)
    model=load_model();model.load_state_dict(torch.load(ROOT/'tiny_model.pt',weights_only=True));model.eval()
    result={}
    for name,ys in [('same_labels_new_phases',original),('unseen_label_sequences',torch.tensor(unseen))]:
        xx=[]
        for y in ys:
            wave,meta=synthesize_sequence(''.join(str(int(v)) for v in y),rng,SynthesisConfig(tone_ms=(180,180),gap_ms=(30,30)),corruption=False)
            xx.append(np.stack([np.pad(wave[e['start_sample']:e['end_sample']],(0,960)) for e in meta['events']]))
        result[name]=diagnostics(model,torch.tensor(np.stack(xx)),ys)
    (ROOT/'tiny_generalization.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{m:v[m] for m in ['sequence_accuracy','digit_accuracy','generated_code_accuracy']} for k,v in result.items()},indent=2))

if __name__=='__main__':main()
