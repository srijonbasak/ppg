"""Decoder hierarchy-use controls and baseline timing; no training."""
import json
import time
import numpy as np
import torch
from ppf.compositional import CompositionalTokenizer
from ppf.sequence import SequencePPF,mixture_scores
from ppf.model import CNNBaseline
from compositional_probes import ROOT,C,CFG,load_data,original_tokenizer

def main():
    torch.set_num_threads(1);data=load_data();x,y,cond=data['test'];result={}
    with torch.inference_mode():
        for name in ['original','structured']:
            if name=='original':tok=original_tokenizer()
            else:
                tok=CompositionalTokenizer(CFG);tok.load_state_dict(torch.load(ROOT/'structured_tokenizer.pt',weights_only=True));tok.freeze()
            model=SequencePPF(tok,CFG);model.load_state_dict(torch.load(ROOT/f'{name}_generator.pt',weights_only=True));model.eval()
            target=tok(y);fine=tok.resize(tok.books[2].lookup(target['codes'][2]),4)
            logits=tok.decoder(fine)
            result[name]={'oracle_fine_only_digit_accuracy':float((logits.argmax(-1)==y).float().mean()),
                          'oracle_fine_only_sequence_accuracy':float((logits.argmax(-1)==y).all(-1).float().mean()),
                          'fine_codes_per_digit':{}}
            allcodes=target['codes'][2]
            for digit in range(10):
                result[name]['fine_codes_per_digit'][str(digit)]=int(allcodes[y==digit].unique().numel())
            _,generated,_=model.generate(x,stochastic=False)
            full=tok.decode_codes(generated).argmax(-1)
            fine=tok.decoder(tok.resize(tok.books[2].lookup(generated[2]),4)).argmax(-1)
            swapped=tok.decode_codes([c.roll(1,0) if l<2 else c for l,c in enumerate(generated)]).argmax(-1)
            result[name].update({'generated_fine_only_digit_accuracy':float((fine==y).float().mean()),
                                'generated_decoder_digits_changed_when_removing_coarse_and_middle':float((full!=fine).float().mean()),
                                'generated_decoder_digits_changed_when_swapping_coarse_and_middle':float((full!=swapped).float().mean())})
        cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.eval()
        for _ in range(3):mixture_scores(cnn(x[0]).softmax(-1)[None,None]).topk(5,-1)
        lat=[];scores=[]
        for sample in x:
            begin=time.perf_counter();s=mixture_scores(cnn(sample).softmax(-1)[None,None]);s.topk(5,-1)
            lat.append((time.perf_counter()-begin)*1000);scores.append(s)
        s=torch.cat(scores);ids=y@torch.tensor([1000,100,10,1]);top=s.topk(5,-1).indices
        result['cnn_baseline']={'sequence_accuracy':float((top[:,0]==ids).float().mean()),
                               'sequence_top5':float((top==ids[:,None]).any(-1).float().mean()),
                               'latency_ms':{'p50':float(np.percentile(lat,50)),'p95':float(np.percentile(lat,95))}}
    (ROOT/'hierarchy_use.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
