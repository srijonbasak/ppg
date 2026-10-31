"""Additional frozen corrupted-test ranking and quantization measurements."""
import json,copy
from pathlib import Path
import torch
from decoder_sensitivity import inputs,predictions
from compositional_probes import load_data,CFG
from hierarchy_contribution import reference
from adapter_alignment import current_adapter
from ppf.model import CNNBaseline
from ppf.safe_gate import RelativeFloorLogMel
from ppf.encoder_study import ConditionedNCM
from ppf.appt import resize

@torch.inference_mode()
def main():
 torch.set_num_threads(1);data=load_data();x,y,_=data['test'];truth=y.flatten();allrows={}
 for seed in [317,419,523]:
  cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.encoder.frontend=RelativeFloorLogMel(CFG,.001);cnn.eval();tok,_,_=reference(seed,cnn)
  z,_=inputs(cnn,current_adapter(seed),x);target=tok(z)['codes'];canonical=[book.lookup(c) for book,c in zip(tok.books,target)]
  rows={}
  for modelname in ['original_joint','original_aligned','recognition']:
   path=Path(f'decoder_results/seed_{seed}/recognition.pt') if modelname=='recognition' else Path(f'alignment_results/seed_{seed}/{modelname}.pt')
   if not path.exists():continue
   model=ConditionedNCM(copy.deepcopy(tok));model.load_state_dict(torch.load(path,weights_only=True));model.eval()
   normal,normal_codes,normal_prefix=predictions(model,z,target,'generated');coarse_correct=normal_codes[0].flatten()==target[0].flatten();out={}
   normal_score=normal.gather(1,truth[:,None]).flatten()
   normal_rank=(normal>normal_score[:,None]).sum(-1)+1
   for mode in ['generated','oracle0','oracle1','oracle2','oracle_all','fixed0','fixed1','fixed2','perturb0','perturb1','perturb2']:
    logits,codes,prefix=predictions(model,z,target,mode);correct=logits.gather(1,truth[:,None]).flatten();rank=(logits>correct[:,None]).sum(-1)+1
    wrong=logits.scatter(1,truth[:,None],-torch.inf).max(-1).values
    v={'correct_logit_mean':float(correct.mean()),'correct_logit_delta':float((correct-normal_score).mean()),'correct_mean_rank':float(rank.float().mean()),'correct_rank_delta':float((rank-normal_rank).float().mean()),'correct_margin_mean':float((correct-wrong).mean()),'code_vector_mse_to_canonical':[float((book.lookup(c)-q).square().mean()) for book,c,q in zip(tok.books,codes,canonical)],'groups':{}}
    for label,mask in [('coarse_correct',coarse_correct),('coarse_incorrect',~coarse_correct)]:
     v['groups'][label]={'n':int(mask.sum()),'digit_accuracy':float((logits.argmax(-1)[mask]==truth[mask]).float().mean()),'rank_delta':float((rank[mask]-normal_rank[mask]).float().mean())} if mask.any() else {'n':0}
    out[mode]=v
   out['generated_prefix_digit_accuracy']=[float((model.ranker(q).argmax(-1)==truth).float().mean()) for q in normal_prefix]
   out['canonical_prefix_quantization_mse']=[float((sum(resize(q,4) for q in canonical[:l+1])-z).square().mean()) for l in range(3)]
   rows[modelname]=out
  allrows[str(seed)]=rows
 Path('decoder_results/detail.json').write_text(json.dumps(allrows,indent=2))
if __name__=='__main__':main()
