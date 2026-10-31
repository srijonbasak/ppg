"""Reversible evidence adapter and parameter-preserving code-alignment controls."""
import torch
from torch import nn
import torch.nn.functional as F
from .appt import resize


class LosslessEvidenceAdapter(nn.Module):
    """Four preserved anchor tokens plus reversible native stem/final packing.

    Native 2×64×31 values become124×32 tokens; ordered time/type offsets are
    fixed and invertible. The NCM receives128×32 memory without new parameters.
    """
    def __init__(self):
        super().__init__();self.register_buffer('mean',torch.zeros(2,64,1));self.register_buffer('std',torch.ones(2,64,1))
        time=torch.arange(31).repeat_interleave(2).repeat(2).float()
        types=torch.arange(4).reshape(2,1,2).expand(2,31,2).reshape(-1).float()
        frequency=torch.exp(torch.arange(0,32,2).float()*(-9.210340372/32))
        phase=time[:,None]*frequency[None]+(types[:,None]+1)*.7
        bias=torch.zeros(124,32);bias[:,0::2]=phase.sin();bias[:,1::2]=phase.cos()
        self.register_buffer('position_bias',.1*bias)

    @torch.no_grad()
    def fit(self,stem,final):
        x=torch.stack([stem,final],1)
        self.mean.copy_(x.mean((0,3)).unsqueeze(-1));self.std.copy_(x.std((0,3)).clamp_min(.01).unsqueeze(-1))

    def forward(self,stem,final,anchors):
        x=(torch.stack([stem,final],1)-self.mean)/self.std
        packed=x.permute(0,1,3,2).reshape(-1,124,32)+self.position_bias
        return torch.cat([anchors,packed],1)

    def unpack(self,memory):
        packed=memory[:,4:]-self.position_bias
        native=packed.reshape(-1,2,31,64).permute(0,1,3,2)*self.std+self.mean
        return native[:,0],native[:,1]


def coarse_summary(stem,final):
    return torch.cat([F.adaptive_avg_pool1d(stem,4).flatten(1),F.adaptive_avg_pool1d(final,4).flatten(1)],-1)


def padded(x,width=512):return F.pad(x.flatten(1),(0,width-x.flatten(1).shape[1]))


class CoarseCodeProbe(nn.Module):
    def __init__(self,mean,std):
        super().__init__();self.register_buffer('mean',mean);self.register_buffer('std',std)
        self.head=nn.Linear(512,64)
    def forward(self,x):return self.head((x-self.mean)/self.std)


def level1_path(model,condition):
    position=model.audio_position if condition.shape[1]==4 else resize(model.audio_position,condition.shape[1])
    acoustic=condition+position
    memory=model.predictor.audio_proj(acoustic)
    query=model.predictor.positions(1)[None].expand(len(condition),-1,-1)+model.predictor.level_embedding.weight[0][None,None]
    hidden=model.predictor.decoder(query,memory)
    logits=model.predictor.heads[0](hidden)
    return {'adapter':condition,'positioned':acoustic,'projected':memory,'hidden':hidden,'logits':logits}


def oracle_rollout(model,condition,oracle_first):
    position=model.audio_position if condition.shape[1]==4 else resize(model.audio_position,condition.shape[1])
    audio=(condition+position).transpose(1,2);levels=[];codes=[];logits=[]
    for l,(n,book) in enumerate(zip(model.tokenizer.lengths,model.tokenizer.books)):
        out=model.predictor.forward_level(audio,levels,l,n)
        c=oracle_first if l==0 else out.argmax(-1)
        levels.append(book.lookup(c));codes.append(c);logits.append(out)
    reconstruction=sum(resize(q,4) for q in levels)
    return {'codes':codes,'digit_logits':model.ranker(reconstruction),'code_logits':logits}


def aligned_loss(model,condition,labels,target):
    """Same update/data budget: code CE trains predictor; observed-code CE trains decoder.

    Digit gradients cannot reward alternative code identities in the predictor.
    No target code is supplied to the generated prediction path.
    """
    out=model(condition,differentiable=False)
    code_loss=sum(F.cross_entropy(logit.flatten(0,1),code.flatten()) for logit,code in zip(out['code_logits'],target))/3
    total=None;digits=[]
    for book,code in zip(model.tokenizer.books,target):
        contribution=resize(book.lookup(code),4)
        total=contribution if total is None else total+contribution
        digits.append(F.cross_entropy(model.ranker(total).reshape(-1,10),labels))
    return code_loss+torch.stack(digits).mean()
