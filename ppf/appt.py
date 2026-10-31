"""Local acoustic Adaptive Parallel Possibility Tree (APPT) prototype.

No complete-sequence label tokenizer is used. The hierarchy represents one
observed sound. This is a small NCM-inspired adaptation using MLP level heads,
not the paper's text Transformer or an official NCM implementation.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .sequence import EMACodebook


def resize(x,n):
    return F.interpolate(x.transpose(1,2),size=n,mode='linear',align_corners=False).transpose(1,2)


def pool(x,n):
    return F.adaptive_avg_pool1d(x.transpose(1,2),n).transpose(1,2)


class LocalAcousticTokenizer(nn.Module):
    lengths=(1,2,4)
    def __init__(self,dim=32,size=64):
        super().__init__()
        self.register_buffer('mean',torch.zeros(64));self.register_buffer('std',torch.ones(64))
        self.register_buffer('projection',torch.eye(64)[:,:dim]);self.register_buffer('scale',torch.ones(dim))
        self.books=nn.ModuleList([EMACodebook(size,dim) for _ in self.lengths])

    @torch.no_grad()
    def fit_projection(self,cnn_features):
        x=pool(cnn_features.transpose(1,2),4).reshape(-1,64)
        self.mean.copy_(x.mean(0));self.std.copy_(x.std(0).clamp_min(.01))
        z=(x-self.mean)/self.std
        _,vectors=torch.linalg.eigh(z.t()@z/(len(z)-1))
        self.projection.copy_(vectors.flip(1)[:,:self.projection.shape[1]])
        self.scale.copy_((z@self.projection).std(0).clamp_min(.01))

    def project(self,features):
        x=pool(features.transpose(1,2),4)
        return (((x-self.mean)/self.std)@self.projection)/self.scale

    def forward(self,z,cutoff=3):
        total=torch.zeros_like(z);codes=[];prefixes=[]
        for l,(n,book) in enumerate(zip(self.lengths,self.books)):
            residual=pool(z-total,n)
            active=book.training
            if self.training and l>=cutoff:book.eval()
            q,c,_=book(residual)
            book.train(active)
            total=total+resize(q,4);codes.append(c);prefixes.append(total)
        return {'codes':codes,'prefixes':prefixes}

    def freeze(self):
        self.eval()
        for p in self.parameters():p.requires_grad_(False)
        return self


class LocalReranker(nn.Module):
    def __init__(self,dim=32):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(4*dim,64),nn.GELU(),nn.Linear(64,10))
    def forward(self,z):return self.head(z.flatten(1))


class LocalHierarchicalRefiner(nn.Module):
    def __init__(self,tokenizer,use_prior=True):
        super().__init__();self.tokenizer=tokenizer.freeze();self.use_prior=use_prior
        dim=tokenizer.projection.shape[1];size=len(tokenizer.books[0].weight)
        self.audio=nn.Linear(dim,dim);self.prior=nn.Linear(dim,dim)
        self.position=nn.Parameter(torch.randn(1,4,dim)*.02)
        self.level=nn.Embedding(3,dim)
        self.hidden=nn.Sequential(nn.Linear(dim,64),nn.GELU())
        self.code_heads=nn.ModuleList([nn.Linear(64,size) for _ in range(3)])
        self.ranker=LocalReranker(dim)

    def train(self,mode=True):
        super().train(mode);self.tokenizer.eval();return self

    def forward(self,z,differentiable=False):
        total=torch.zeros_like(z);prefix_logits=[];logits=[];codes=[]
        for l,(n,book) in enumerate(zip(self.tokenizer.lengths,self.tokenizer.books)):
            query=self.audio(pool(z,n))+self.position[:,:n]+self.level.weight[l]
            if self.use_prior:query=query+self.prior(pool(total,n))
            logit=self.code_heads[l](self.hidden(query));c=logit.argmax(-1)
            if differentiable:
                p=logit.softmax(-1);onehot=F.one_hot(c,logit.shape[-1]).to(p.dtype)
                w=onehot+(p-p.detach());q=w@book.weight
            else:q=book.lookup(c)
            total=total+resize(q,4)
            # All ten local digit-candidate scores are computed together.
            prefix_logits.append(self.ranker(total));logits.append(logit);codes.append(c)
        return {'digit_logits':prefix_logits,'code_logits':logits,'codes':codes}


def uncertainty_gate(base_probs,verification_probs,confidence_threshold,audit_threshold=.8):
    confidence,pred=base_probs.max(-1)
    uncertain=confidence<confidence_threshold
    vconf,vpred=verification_probs.max(-1)
    reopened=(~uncertain)&(vpred!=pred)&(vconf>=audit_threshold)
    return uncertain|reopened,reopened


def possibility_tree(probs,expand,top_k=5):
    """Exact conditional search; fixed positions remain provisional.

    Returned probabilities are full factorized model scores. Search mass makes
    omitted alternatives visible; no claim of an exhaustive posterior is made.
    """
    choices=[torch.arange(10,device=probs.device) if bool(expand[i]) else probs[i].argmax().reshape(1) for i in range(4)]
    candidates=torch.cartesian_prod(*choices).reshape(-1,4)
    score=torch.ones(len(candidates),device=probs.device)
    for i in range(4):score*=probs[i,candidates[:,i]]
    values,indices=score.topk(min(top_k,len(score)))
    return {'sequences':candidates[indices],'probabilities':values,'branches':len(candidates),
            'retained_probability_mass':float(score.sum())}


class AdaptivePossibilityTree:
    def __init__(self,cnn,tokenizer,verifier,refiner,temperatures,threshold,audit_threshold=.8):
        self.cnn=cnn.eval();self.tokenizer=tokenizer.freeze();self.verifier=verifier.eval();self.refiner=refiner.eval()
        self.temperatures=temperatures;self.threshold=threshold;self.audit_threshold=audit_threshold

    @torch.inference_mode()
    def predict(self,events,method='appt',top_k=5,base_override=None):
        if method not in ('cnn','cnn_adaptive','local','appt'):
            raise ValueError('Unknown prediction method')
        # Four oracle event windows; no labels or target codes are accepted.
        features=self.cnn.encoder(events)
        base=(self.cnn.head(features)/self.temperatures['cnn']).softmax(-1)
        if base_override is not None:base=base_override.clone()
        erased=events.square().mean(-1)<=1e-12
        base[erased]=.1
        if method=='cnn':
            expand=(base.max(-1).values<self.threshold)|erased
            tree=possibility_tree(base,torch.ones_like(expand),top_k)
            return {'digit_probabilities':base,'expanded':expand,'reopened':torch.zeros_like(expand),
                    'refined_positions':0,'erased':erased,**tree}
        z=self.tokenizer.project(features)
        verification=(self.verifier(z)/self.temperatures['local']).softmax(-1)
        # Explicit no-evidence rule, shared across every comparison method.
        base[erased]=.1;verification[erased]=.1
        expand,reopened=uncertainty_gate(base,verification,self.threshold,self.audit_threshold)
        expand=expand|erased
        result=base.clone();chosen=expand&(~erased)
        if method=='local' and chosen.any():result[chosen]=verification[chosen]
        elif method=='appt' and chosen.any():
            logits=self.refiner(z[chosen])['digit_logits'][-1]
            result[chosen]=(logits/self.temperatures['appt']).softmax(-1)
        result[erased]=.1
        tree=possibility_tree(result,torch.ones_like(expand) if method=='cnn' else expand,top_k)
        return {'digit_probabilities':result,'expanded':expand,'reopened':reopened,
                'refined_positions':int(chosen.sum()) if method in ('local','appt') else 0,
                'erased':erased,**tree}


def load_appt(checkpoint_directory,ablation=False):
    """Load the saved local research prototype; input remains four tone windows."""
    import json
    from pathlib import Path
    from .config import ModelConfig
    from .model import CNNBaseline
    root=Path(checkpoint_directory);metadata=json.loads((root/'metrics.json').read_text())
    payload=torch.load(root/'cnn.pt',weights_only=True)
    cnn=CNNBaseline(ModelConfig.from_dict(payload['config']));cnn.load_state_dict(payload['model'])
    config=metadata['config'];tok=LocalAcousticTokenizer(config['latent_dim'],config['codebook_size'])
    tok.load_state_dict(torch.load(root/'acoustic_tokenizer.pt',weights_only=True));tok.freeze()
    verifier=LocalReranker(config['latent_dim']);verifier.load_state_dict(torch.load(root/'local.pt',weights_only=True))
    refiner=LocalHierarchicalRefiner(tok,use_prior=not ablation)
    refiner.load_state_dict(torch.load(root/('no_prior.pt' if ablation else 'appt.pt'),weights_only=True))
    temperatures=metadata['temperatures'].copy()
    if ablation:temperatures['appt']=temperatures['no_prior']
    return AdaptivePossibilityTree(cnn,tok,verifier,refiner,temperatures,metadata['confidence_threshold'],config['audit_confidence_threshold'])
