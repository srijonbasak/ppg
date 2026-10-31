"""Separate code generation, oracle-prefix feedback and latent decoding."""
import json
from pathlib import Path
import torch
from ppf.model import CNNBaseline
from ppf.safe_gate import RelativeFloorLogMel
from ppf.encoder_study import ChannelProjection,ConditionedNCM,condition_input
from ppf.appt import resize
from compositional_probes import CFG,load_data
from hierarchy_contribution import reference
from encoder_information import ROOT,C,extract,missing_frequency
import appt_experiment as ap


def main():
    torch.set_num_threads(1);d=json.loads((ROOT/'metrics.json').read_text());data=load_data()
    noise=torch.tensor([v=='noise' for v in data['test'][2]]);partial=torch.tensor([v=='partial_erasure' for v in data['test'][2]])
    scenarios={'corrupted':(data['test'][0],data['test'][1]),'clean':(ap.scenarios(data)['clean'][0],data['test'][1]),
        'noisy':(data['test'][0][noise],data['test'][1][noise]),'partial_erasure':(data['test'][0][partial],data['test'][1][partial]),
        'missing_frequency':(missing_frequency(data['test'][1],818),data['test'][1])}
    for seed,saved in d['seeds'].items():
        folder=ROOT/f'seed_{seed}';cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.encoder.frontend=RelativeFloorLogMel(CFG,.001);cnn.eval()
        tok,_,_=reference(int(seed),cnn);states=torch.load(folder/'adapters.pt',weights_only=True);adapters={}
        for k,state in states.items():
            a=ChannelProjection(len(state['mean']));a.load_state_dict(state);adapters[k]=a
        taps={s:extract(cnn,x) for s,(x,y) in scenarios.items()}
        with torch.inference_mode():
            saved['adapter_consistency']={}
            for s,v in taps.items():
                final=condition_input(v,'final',adapters);dense=condition_input(v,'final31',adapters)
                pooled=torch.nn.functional.adaptive_avg_pool1d(dense.transpose(1,2),4).transpose(1,2)
                saved['adapter_consistency'][s]={'dense_pool_max_difference':float((pooled-final).abs().max()),
                    'legacy_projection_max_difference':float((tok.project(v['final'])-final).abs().max())}
        for mode,row in saved['ncm'].items():
            model=ConditionedNCM(tok);model.load_state_dict(torch.load(folder/f'{mode}_ncm.pt',weights_only=True));model.eval()
            row['code_diagnostics']={}
            for s,(x,y) in scenarios.items():
                with torch.inference_mode():
                    memory=condition_input(taps[s],mode,adapters);tokens=tok(condition_input(taps[s],'final',adapters))
                    targets=tokens['codes'];hits=torch.zeros(3);total=torch.zeros(3)
                    for offset in range(0,len(memory),128):
                        condition=memory[offset:offset+128];codes=[v[offset:offset+128] for v in targets]
                        position=model.audio_position if condition.shape[1]==4 else resize(model.audio_position,condition.shape[1])
                        audio=(condition+position).transpose(1,2)
                        previous=[]
                        for l,(n,book,code) in enumerate(zip(tok.lengths,tok.books,codes)):
                            logits=model.predictor.forward_level(audio,previous,l,n)
                            hits[l]+=(logits.argmax(-1)==code).sum();total[l]+=code.numel();previous.append(book.lookup(code))
                    observed=model.ranker(tokens['prefixes'][-1]).argmax(-1).reshape(-1,4)
                    row['code_diagnostics'][s]={'oracle_prefix_code_accuracy':(hits/total).tolist(),
                        'observed_code_decoder_digit_accuracy':float((observed==y).float().mean()),
                        'observed_code_decoder_sequence_accuracy':float((observed==y).all(-1).float().mean())}
        # Raw per-recording digit-probe outcomes support paired bootstrap.
        with torch.inference_mode():
            for stage,probe_meta in saved['digit_probes'].items():
                from ppf.encoder_study import DigitProbe,digit_input
                probe=DigitProbe(torch.zeros(256),torch.ones(256));probe.load_state_dict(torch.load(folder/f'{stage}_digit_probe.pt',weights_only=True));probe.eval()
                for s,(x,y) in scenarios.items():
                    p=(probe(digit_input(taps[s],stage,adapters))/probe_meta['temperature']).softmax(-1)
                    ok=p.argmax(-1).reshape(-1,4)==y
                    (folder/f'rawprobe_{stage}_{s}_trace.json').write_text(json.dumps([{'label':v.tolist(),'digit_correct':c.int().tolist()} for v,c in zip(y,ok)]))
        (ROOT/'metrics.json').write_text(json.dumps(d,indent=2));print(seed,'code/oracle/adapter diagnostics saved',flush=True)

if __name__=='__main__':main()
