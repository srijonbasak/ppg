"""Explicit oracle diagnostics and recognition-aware generated-code training."""
import torch
import torch.nn.functional as F
from .appt import resize


def intervene(model,z,targets=None,replace=(),fixed=None,perturb=None):
    audio=(z+model.audio_position).transpose(1,2);levels=[];codes=[];prefix=[]
    for l,(n,book) in enumerate(zip(model.tokenizer.lengths,model.tokenizer.books)):
        logits=model.predictor.forward_level(audio,levels,l,n)
        c=(fixed[l] if fixed is not None else logits.argmax(-1)).clone()
        if l in replace:c=targets[l]
        if perturb==l:c[:,0]=(c[:,0]+1)%64
        codes.append(c);levels.append(book.lookup(c));prefix.append(sum(resize(v,4) for v in levels))
    return {'codes':codes,'prefix':prefix,'logits':model.ranker(prefix[-1])}


def recognition_loss(model,z,y,target,rank_mask):
    """Hard generated forward, straight-through soft gradient; no oracle prefix."""
    out=model(z,differentiable=True)
    code=sum(F.cross_entropy(a.flatten(0,1),b.flatten()) for a,b in zip(out['code_logits'],target))/3
    digit=sum(F.cross_entropy(a,y) for a in out['digit_logits'])/3
    ranking=[]
    for a in out['digit_logits']:
        correct=a.gather(1,y[:,None]);wrong=a.masked_fill(F.one_hot(y,10).bool(),-torch.inf).max(1,keepdim=True).values
        terms=F.softplus(1+wrong-correct).flatten()
        ranking.append(terms[rank_mask].mean() if rank_mask.any() else terms.sum()*0)
    return .1*code+digit+.25*torch.stack(ranking).mean()
