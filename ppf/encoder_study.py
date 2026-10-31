"""Read-only encoder taps and fixed-width conditioning adapters."""
import torch
from torch import nn
import torch.nn.functional as F
from .graph_ncm import AcousticNCMTransformer
from .appt import resize


def missing_frequency(labels,seed):
    """Remove column-frequency identity without leaking it through timing/phase."""
    import numpy as np
    from .data import DTMF_FREQUENCIES
    rng=np.random.default_rng(seed);all_events=[]
    for seq in labels:
        events=[]
        for digit in seq:
            f=DTMF_FREQUENCIES[str(int(digit))][0];t=np.arange(1440,dtype=np.float32)/8000
            tone=(.36*np.sin(2*np.pi*f*t+rng.uniform(0,2*np.pi))).astype(np.float32)
            ramp=np.linspace(0,1,40,dtype=np.float32);tone[:40]*=ramp;tone[-40:]*=ramp[::-1]
            events.append(np.pad(tone,(0,960)))
        all_events.append(np.stack(events))
    return torch.tensor(np.stack(all_events))


def encoder_taps(cnn,events,stop=None):
    result={};x=cnn.encoder.frontend(events);result['mel']=x
    if stop=='mel':return result
    x=cnn.encoder.stem(x);result['stem']=x
    if stop=='stem':return result
    for i,block in enumerate(cnn.encoder.blocks):
        x=block(x);key=['block1','block2','final'][i];result[key]=x
        if stop==key:return result
    return result


class ChannelProjection(nn.Module):
    def __init__(self,channels):
        super().__init__();self.register_buffer('mean',torch.zeros(channels));self.register_buffer('std',torch.ones(channels))
        self.register_buffer('projection',torch.eye(channels)[:,:32]);self.register_buffer('scale',torch.ones(32))
    @torch.no_grad()
    def fit(self,features):
        x=F.adaptive_avg_pool1d(features,4).transpose(1,2).reshape(-1,len(self.mean))
        self.mean.copy_(x.mean(0));self.std.copy_(x.std(0).clamp_min(.01));z=(x-self.mean)/self.std
        _,v=torch.linalg.eigh(z.t()@z/(len(z)-1));self.projection.copy_(v.flip(1)[:,:32]);self.scale.copy_((z@self.projection).std(0).clamp_min(.01))
    def frames(self,features):return (((features.transpose(1,2)-self.mean)/self.std)@self.projection)/self.scale
    def forward(self,features):return F.adaptive_avg_pool1d(self.frames(features).transpose(1,2),4).transpose(1,2)


def stage_features(taps,name):
    if name=='multiscale':return torch.cat([taps[k] for k in ['stem','block1','block2','final']],1)
    if name.endswith('_pca'):return taps[name[:-4]]
    return taps['final'] if name=='final_mean' else taps[name]


def digit_input(taps,name,adapters):
    if name.endswith('_pca') or name=='multiscale':
        x=adapters[name[:-4] if name.endswith('_pca') else name](stage_features(taps,name))
        return F.pad(x,(0,32)).flatten(1)
    x=stage_features(taps,name)
    if name=='final_mean':x=x.mean(-1,keepdim=True).expand(-1,-1,4)
    else:x=F.adaptive_avg_pool1d(x,4)
    x=x.transpose(1,2)
    return F.pad(x,(0,64-x.shape[-1])).flatten(1)


def condition_input(taps,name,adapters):
    if name=='final31':return adapters['final'].frames(taps['final'])
    return adapters[name](stage_features(taps,name))


class DigitProbe(nn.Module):
    def __init__(self,mean,std):
        super().__init__();self.register_buffer('mean',mean);self.register_buffer('std',std)
        self.net=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,10))
    def forward(self,x):return self.net((x-self.mean)/self.std)


class ConditionedNCM(AcousticNCMTransformer):
    """Same parameters/decoder/targets; cross-attention memory may have 4 or 31 tokens."""
    def forward(self,z,differentiable=False):
        position=self.audio_position if z.shape[1]==4 else resize(self.audio_position,z.shape[1])
        audio=(z+position).transpose(1,2)
        levels=[];codes=[];code_logits=[];digit_logits=[]
        for l,(n,book) in enumerate(zip(self.tokenizer.lengths,self.tokenizer.books)):
            logits=self.predictor.forward_level(audio,levels if self.use_prior else [],l,n);c=logits.argmax(-1)
            if differentiable:
                p=logits.softmax(-1);w=F.one_hot(c,64).to(p.dtype)+(p-p.detach());q=w@book.weight
            else:q=book.lookup(c)
            levels.append(q);codes.append(c);code_logits.append(logits)
            digit_logits.append(self.ranker(sum(resize(v,4) for v in levels)))
        return {'codes':codes,'code_logits':code_logits,'digit_logits':digit_logits}
