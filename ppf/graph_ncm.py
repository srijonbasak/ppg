"""Minimal acoustic relation graph and graph-conditioned NCM research model.

42 nodes: 4 local CNN-time nodes, 28 time/frequency evidence nodes and 10
candidate nodes. Fixed DTMF incidence defines evidence-to-candidate edges.
Message aggregation is fixed; the projection/codebooks and NCM are learned.
This is deliberately not a large trainable GNN or a whole-sequence graph.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .data import DTMF_FREQUENCIES
from .appt import LocalReranker,resize
from .model import ParallelLevelPredictor
from .config import ModelConfig

FREQUENCIES=(697,770,852,941,1209,1336,1477)
PAIRS=tuple(tuple(FREQUENCIES.index(f) for f in DTMF_FREQUENCIES[str(d)]) for d in range(10))


def frequency_evidence(events):
    """Four measured local spectra; no labels or corruption flags."""
    chunks=events.reshape(-1,4,600)
    window=torch.hann_window(600,device=events.device)
    power=torch.fft.rfft(chunks*window,n=2048).abs().square()
    grid=torch.arange(power.shape[-1],device=events.device)*8000/2048
    values=[]
    for f in FREQUENCIES:
        mask=(grid>=f*.975)&(grid<=f*1.025)
        values.append(power[:,:,mask].amax(-1))
    peaks=torch.stack(values,-1)
    return (peaks/power.sum(-1,keepdim=True).clamp_min(1e-12)).clamp_min(1e-9).log()


def graph_messages(evidence,pairs=PAIRS):
    # Adjacent time evidence has fixed continuity edges within the local tone.
    pad=F.pad(evidence.transpose(1,2),(1,1),mode='replicate').transpose(1,2)
    smoothed=.25*pad[:,:-2]+.5*pad[:,1:-1]+.25*pad[:,2:]
    links=torch.as_tensor(pairs,device=evidence.device)
    # A digit requires both incident frequencies; one strong peak alone is
    # insufficient. These are edge messages, not ground-truth digit inputs.
    return torch.minimum(smoothed[:,:,links[:,0]],smoothed[:,:,links[:,1]])


class EvidenceProjection(nn.Module):
    def __init__(self,graph=False,rewired=False):
        super().__init__();self.graph=graph;self.rewired=rewired
        dim=49 if graph else 39
        self.register_buffer('mean',torch.zeros(dim));self.register_buffer('std',torch.ones(dim))
        self.register_buffer('projection',torch.eye(dim)[:,:32]);self.register_buffer('scale',torch.ones(32))
        pairs=torch.tensor(PAIRS)
        if rewired:
            order=torch.randperm(10,generator=torch.Generator().manual_seed(1201))
            pairs[:,1]=pairs[order,1].clone()
        self.register_buffer('pairs',pairs)

    def raw(self,cnn_local,evidence):
        values=[cnn_local,evidence]
        if self.graph:values.append(graph_messages(evidence,self.pairs))
        return torch.cat(values,-1)

    @torch.no_grad()
    def fit(self,cnn_local,evidence):
        raw=self.raw(cnn_local,evidence).reshape(-1,len(self.mean))
        self.mean.copy_(raw.mean(0));self.std.copy_(raw.std(0).clamp_min(.01))
        z=(raw-self.mean)/self.std;_,vectors=torch.linalg.eigh(z.t()@z/(len(z)-1))
        self.projection.copy_(vectors.flip(1)[:,:32]);self.scale.copy_((z@self.projection).std(0).clamp_min(.01))

    def forward(self,cnn_local,evidence):
        return (((self.raw(cnn_local,evidence)-self.mean)/self.std)@self.projection)/self.scale


class AcousticNCMTransformer(nn.Module):
    def __init__(self,tokenizer,use_prior=True):
        super().__init__();self.tokenizer=tokenizer.freeze();self.use_prior=use_prior
        cfg=ModelConfig(model_dim=32,code_dim=32,codebook_size=64,transformer_layers=1,transformer_heads=4)
        self.predictor=ParallelLevelPredictor(cfg)
        self.audio_position=nn.Parameter(torch.randn(1,4,32)*.02)
        self.ranker=LocalReranker()

    def train(self,mode=True):
        super().train(mode);self.tokenizer.eval();return self

    def forward(self,z,differentiable=False):
        audio=(z+self.audio_position).transpose(1,2)
        levels=[];codes=[];code_logits=[];digit_logits=[]
        for l,(n,book) in enumerate(zip(self.tokenizer.lengths,self.tokenizer.books)):
            logits=self.predictor.forward_level(audio,levels if self.use_prior else [],l,n)
            c=logits.argmax(-1)
            if differentiable:
                p=logits.softmax(-1);w=F.one_hot(c,64).to(p.dtype)+(p-p.detach())
                q=w@book.weight
            else:q=book.lookup(c)
            levels.append(q);codes.append(c);code_logits.append(logits)
            digit_logits.append(self.ranker(sum(resize(v,4) for v in levels)))
        return {'codes':codes,'code_logits':code_logits,'digit_logits':digit_logits}

    def loss(self,z,labels,target):
        out=self(z,differentiable=True)
        return sum(F.cross_entropy(v,labels) for v in out['digit_logits'])/3+.1*sum(
            F.cross_entropy(v.flatten(0,1),t.flatten()) for v,t in zip(out['code_logits'],target))/3


class CNNIdentityProjection(nn.Module):
    """Retain the existing CNN-only acoustic embedding as a strong control."""
    def forward(self,cnn_local,evidence=None):return cnn_local


class GraphConditionedTree:
    """Audio-only research inference using the preserved conventional gate."""
    def __init__(self,cnn,local_tokenizer,verifier,gate,gate_metadata,projection,model,ncm,temperature):
        self.cnn=cnn.eval();self.local_tokenizer=local_tokenizer.freeze();self.verifier=verifier.eval()
        self.gate=gate.eval();self.meta=gate_metadata;self.projection=projection.eval();self.model=model.eval()
        self.ncm=ncm;self.temperature=temperature

    @torch.inference_mode()
    def predict(self,events,top_k=5,guard_missing_band=True):
        from .safe_gate import gate_features
        from .appt import possibility_tree
        features=self.cnn.encoder(events);local=self.local_tokenizer.project(features)
        p=(self.cnn.head(features)/self.meta['temperatures']['cnn']).softmax(-1)
        proxy=(self.verifier(local)/self.meta['temperatures']['local']).softmax(-1)
        erased=events.square().mean(-1)<=1e-12;p[erased]=.1;proxy[erased]=.1
        spectral=frequency_evidence(events)
        band_mask,band_probs=missing_band_distribution(events,spectral)
        if not guard_missing_band:band_mask=torch.zeros_like(band_mask)
        rms=events.square().mean(-1).sqrt()
        route=(self.gate.score(gate_features(p,proxy,rms,local))>=self.meta['route_threshold'])|erased|band_mask
        accepted=route&(~erased)&(~band_mask)&(self.gate.score(gate_features(p,proxy,rms,local))>=self.meta['accept_threshold'])
        result=p.clone()
        if accepted.any():
            z=self.projection(local[accepted],spectral[accepted])
            logits=self.model(z)['digit_logits'][-1] if self.ncm else self.model(z)
            result[accepted]=(logits/self.temperature).softmax(-1)
        result[erased]=.1
        result[band_mask]=band_probs[band_mask]
        return {'digit_probabilities':result,'expanded':route,'accepted':accepted,
                'missing_band_guarded':band_mask,
                **possibility_tree(result,route,top_k)}


def load_graph_tree(directory,seed=317,variant='D',budget='matched_flops'):
    """Load graph+Transformer by default; other variants are explicit controls."""
    import json
    from pathlib import Path
    from .model import CNNBaseline
    from .appt import LocalAcousticTokenizer
    from .safe_gate import RelativeFloorLogMel,BenefitGate
    root=Path(directory);repo=root.parent;folder=root/f'seed_{seed}'
    metrics=json.loads((root/'metrics.json').read_text())['seeds'][str(seed)]
    meta=json.loads((repo/'gate_results/metrics.json').read_text())['seeds'][str(seed)]
    cnn=CNNBaseline();cnn.load_state_dict(torch.load(repo/'representation_results/cnn.pt',weights_only=True))
    cnn.encoder.frontend=RelativeFloorLogMel(cnn.cfg,.001)
    gate_folder=repo/'gate_results'/f'seed_{seed}'
    tok=LocalAcousticTokenizer();tok.load_state_dict(torch.load(gate_folder/'acoustic_tokenizer.pt',weights_only=True));tok.freeze()
    verifier=LocalReranker();verifier.load_state_dict(torch.load(gate_folder/'local.pt',weights_only=True))
    gate=BenefitGate();gate.load_state_dict(torch.load(gate_folder/'gate.pt',weights_only=True))
    ncm=variant in ('C','D','C_cnn','D_no_prior')
    if variant.endswith('_cnn'):
        projection=CNNIdentityProjection();code_tok=tok
    else:
        name='rewired' if variant=='rewired' else 'graph' if variant in ('B','D','D_no_prior') else 'flat'
        projection=EvidenceProjection(name!='flat',name=='rewired');projection.load_state_dict(torch.load(folder/f'{name}_projection.pt',weights_only=True))
        code_tok=LocalAcousticTokenizer();code_tok.load_state_dict(torch.load(folder/f'{name}_tokenizer.pt',weights_only=True))
    model=AcousticNCMTransformer(code_tok,variant!='D_no_prior') if ncm else LocalReranker()
    checkpoint=folder/f'{variant}_{budget}.pt'
    if not checkpoint.exists():checkpoint=folder/f'{variant}_equal_updates.pt'
    model.load_state_dict(torch.load(checkpoint,weights_only=True))
    return GraphConditionedTree(cnn,tok,verifier,gate,meta,projection,model,ncm,metrics['variants'][variant][budget]['temperature'])


def missing_band_distribution(events,evidence=None):
    """Conservative analytic rule for an almost absent DTMF frequency group.

    Assumes a known 0–9 DTMF event with one group removed. It is shared by all
    controls, and does not claim neural recovery of missing information.
    Weak/noisy absent bands may not trigger it. All ten digits retain support.
    """
    energy=events.reshape(-1,4,600).square().mean(-1)
    active=energy>=.8*energy.amax(-1,keepdim=True)
    measured=(frequency_evidence(events) if evidence is None else evidence).exp()
    # Exclude low-energy boundary/padding frames whose truncation leakage can
    # look like a weak second band. No event-boundary metadata is consulted.
    power=(measured*active[:,:,None]).sum(1)/active.sum(1,keepdim=True).clamp_min(1)
    low,li=power[:,:4].max(-1);high,hi=power[:,4:].max(-1)
    low_only=(low>.01)&(high<low*1e-5)
    high_only=(high>.01)&(low<high*1e-5)
    mask=low_only|high_only
    p=torch.full((len(events),10),.1,device=events.device)
    pairs=torch.tensor(PAIRS,device=events.device)
    for i in mask.nonzero().flatten():
        compatible=(pairs[:,0]==li[i]) if low_only[i] else (pairs[:,1]==hi[i]+4)
        count=int(compatible.sum());p[i].fill_(1e-6);p[i,compatible]=(1-(10-count)*1e-6)/count
    return mask,p
