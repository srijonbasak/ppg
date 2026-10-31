"""Evidence-limited diagnostics, with no hidden digit labels at inference."""
import json,math
from pathlib import Path
import torch
from ppf.graph_ncm import load_graph_tree

ROOT=Path('graph_results')

def main():
    torch.set_num_threads(1);result={};raw_result={}
    for seed in [317,419,523]:
        rows={};raw_rows={}
        for variant in ['A_cnn','B','C_cnn','D']:
            model=load_graph_tree(ROOT,seed,variant)
            erased=model.predict(torch.zeros(4,2400))['digit_probabilities']
            values=[];raw_values=[]
            for row,freq in enumerate([697,770,852]):
                compatible=list(range(1+3*row,4+3*row))
                for phase in [0.,1.,2.]:
                    t=torch.arange(1440)/8000;tone=.36*torch.sin(2*math.pi*freq*t+phase)
                    ramp=torch.linspace(0,1,40);tone[:40]*=ramp;tone[-40:]*=ramp.flip(0)
                    events=torch.nn.functional.pad(tone,(0,960)).repeat(4,1)
                    p=model.predict(events)['digit_probabilities']
                    raw=model.predict(events,guard_missing_band=False)['digit_probabilities']
                    raw_values.append({'max_probability':float(raw.max(-1).values.mean()),'compatible_probability_mass':float(raw[:,compatible].sum(-1).mean())})
                    values.append({'row_hz':freq,'phase':phase,'max_probability':float(p.max(-1).values.mean()),
                      'compatible_probability_mass':float(p[:,compatible].sum(-1).mean()),
                      'uniform_compatible_cross_entropy':float(-p[:,compatible].clamp_min(1e-30).log().mean())})
            rows[variant]={'zero_evidence_max_deviation_from_uniform':float((erased-.1).abs().max()),
                'row_only_mean_max_probability':sum(v['max_probability'] for v in values)/len(values),
                'row_only_compatible_mass':sum(v['compatible_probability_mass'] for v in values)/len(values),'details':values}
            raw_rows[variant]={'row_only_mean_max_probability':sum(v['max_probability'] for v in raw_values)/len(raw_values),
                              'row_only_compatible_mass':sum(v['compatible_probability_mass'] for v in raw_values)/len(raw_values),'details':raw_values}
        result[str(seed)]=rows
        raw_result[str(seed)]=raw_rows
    (ROOT/'uncertainty_checks.json').write_text(json.dumps(result,indent=2))
    (ROOT/'uncertainty_checks_unguarded.json').write_text(json.dumps(raw_result,indent=2))
    print(json.dumps({s:{k:(v['row_only_mean_max_probability'],v['zero_evidence_max_deviation_from_uniform']) for k,v in rows.items()} for s,rows in result.items()},indent=2))
    # Exercise the default graph+NCM API on a real, unlabeled inference input.
    data=torch.load('representation_results/audio_data.pt',weights_only=False)
    model=load_graph_tree(ROOT);x=data['test'][0][0];out=model.predict(x)
    (ROOT/'example_prediction.json').write_text(json.dumps({k:v.tolist() if isinstance(v,torch.Tensor) else v for k,v in out.items()},indent=2))

if __name__=='__main__':main()
