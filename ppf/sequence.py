"""Staged clean-label 1/2/4 residual-VQ sequence model.

Reuses the prototype audio CNN and parallel level predictor. EMA codebooks are
buffers: no codebook optimizer gradients are expected. This is an audio
adaptation, not a reproduction of the original NCM training recipe.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .config import ModelConfig
from .model import AcousticEncoder, ParallelLevelPredictor, MultiscaleResidualVQ


class EMACodebook(nn.Module):
    def __init__(self, size=64, dim=64, decay=.95):
        super().__init__()
        self.decay = decay
        self.register_buffer('weight', torch.randn(size, dim) * .1)
        self.register_buffer('counts', torch.zeros(size))
        self.register_buffer('sums', torch.zeros(size, dim))
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('replacements', torch.tensor(0))

    def lookup(self, codes):
        return F.embedding(codes, self.weight)

    @torch.no_grad()
    def initialize(self, flat):
        centers = flat[torch.arange(len(self.weight), device=flat.device) % len(flat)].clone()
        # K-means startup, rather than tiny codes far from the target manifold.
        for _ in range(8):
            idx = torch.cdist(flat, centers).argmin(-1)
            cnt = torch.bincount(idx, minlength=len(centers)).float()
            sums = torch.zeros_like(centers).index_add_(0, idx, flat)
            centers = torch.where(cnt[:, None] > 0, sums / cnt[:, None].clamp_min(1), centers)
        self.weight.copy_(centers); self.counts.fill_(1); self.sums.copy_(centers)
        self.initialized.fill_(True)

    def forward(self, x):
        flat = x.detach().reshape(-1, x.shape[-1])
        if self.training and not bool(self.initialized): self.initialize(flat)
        idx = torch.cdist(flat, self.weight).argmin(-1)
        q = self.lookup(idx).view_as(x)
        error = F.mse_loss(x, q.detach())
        if self.training:
            with torch.no_grad():
                cnt = torch.bincount(idx, minlength=len(self.weight)).float()
                sums = torch.zeros_like(self.weight).index_add_(0, idx, flat)
                self.counts.lerp_(cnt, 1-self.decay)
                self.sums.lerp_(sums, 1-self.decay)
                dead = self.counts < .1
                if dead.any():
                    fresh = flat[torch.randint(len(flat), (int(dead.sum()),), device=flat.device)]
                    self.counts[dead] = 1
                    self.sums[dead] = fresh
                    self.replacements.add_(int(dead.sum()))
                self.weight.copy_(self.sums / self.counts[:, None].clamp_min(1e-5))
        return q, idx.view(x.shape[:-1]), error


class LabelDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.positions = nn.Parameter(torch.randn(1,4,dim)*.02)
        self.net = nn.Sequential(nn.Linear(dim,dim), nn.GELU(), nn.Linear(dim,10))

    def forward(self, latent):
        return self.net(latent + self.positions)


class CleanLabelTokenizer(nn.Module):
    lengths = (1,2,4)
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        dim=self.cfg.code_dim
        self.embedding = nn.Embedding(10,dim)
        self.positions = nn.Parameter(torch.randn(1,4,dim)*.02)
        self.encoder = nn.Sequential(nn.Linear(dim,dim), nn.GELU(), nn.Linear(dim,dim), nn.LayerNorm(dim))
        self.books = nn.ModuleList([EMACodebook(self.cfg.codebook_size,dim) for _ in self.lengths])
        self.decoder = LabelDecoder(dim)

    @staticmethod
    def resize(x,n): return MultiscaleResidualVQ._resize(x,n)

    def forward(self, labels, prefix=None):
        target=self.encoder(self.embedding(labels)+self.positions)
        levels=[]; codes=[]; prefixes=[]; errors=[]
        full=torch.zeros_like(target)
        for n,book in zip(self.lengths,self.books):
            residual=self.resize(target,n)-self.resize(full.detach(),n)
            q,idx,error=book(residual)
            levels.append(q); codes.append(idx); errors.append(error)
            full=full+self.resize(q,4)
            # Straight-through reconstruction; decoder sees discrete codes only.
            prefixes.append(target+(full-target).detach())
        if prefix is None:
            prefix=int(torch.randint(1,4,()).item()) if self.training else 3
        logits=[self.decoder(p) for p in prefixes]
        final_loss=F.cross_entropy(logits[-1].reshape(-1,10),labels.flatten())
        nested_loss=F.cross_entropy(logits[prefix-1].reshape(-1,10),labels.flatten())
        loss=final_loss+nested_loss+.25*torch.stack(errors).mean()
        return {'codes':codes,'levels':levels,'logits':logits,'loss':loss,
                'quantization_error':torch.stack(errors), 'prefix':prefix}

    def freeze(self):
        self.eval()
        for p in self.parameters(): p.requires_grad_(False)
        return self

    def decode_codes(self, codes):
        return self.decoder(sum(self.resize(b.lookup(c),4) for b,c in zip(self.books,codes)))


class SequencePPF(nn.Module):
    def __init__(self, tokenizer, cfg=None):
        super().__init__()
        self.cfg=cfg or tokenizer.cfg
        self.tokenizer=tokenizer.freeze()
        self.encoder=AcousticEncoder(self.cfg)
        self.event_positions=nn.Parameter(torch.randn(1,4,self.cfg.model_dim,1)*.02)
        self.predictor=ParallelLevelPredictor(self.cfg)

    def train(self, mode=True):
        super().train(mode)
        self.tokenizer.eval()
        return self

    def audio_memory(self, events):
        b=events.shape[0]
        encoded=self.encoder(events.reshape(-1,events.shape[-1]))
        encoded=encoded.reshape(b,4,self.cfg.model_dim,-1)+self.event_positions
        return encoded.permute(0,2,1,3).flatten(2)

    def teacher_logits(self, events, codes):
        return self.training_logits(events, codes)

    def training_logits(self, events, codes, generated_prefix_probability=0.):
        """Optional mixed-prefix training; generated codes are detached argmax.

        Targets remain canonical tokenizer codes. This is a robustness
        intervention, not a guarantee that a wrong coarse code is recoverable.
        """
        if not 0 <= generated_prefix_probability <= 1:
            raise ValueError('generated_prefix_probability must be in [0, 1]')
        memory=self.audio_memory(events)
        previous=[]; logits=[]
        for level,(n,book,target) in enumerate(zip(self.tokenizer.lengths,self.tokenizer.books,codes)):
            out=self.predictor.forward_level(memory,previous,level,n)
            logits.append(out)
            if generated_prefix_probability == 0:
                chosen=target
            elif generated_prefix_probability == 1:
                chosen=out.detach().argmax(-1)
            else:
                use_generated=torch.rand(len(events),1,device=events.device)<generated_prefix_probability
                chosen=torch.where(use_generated,out.detach().argmax(-1),target)
            previous.append(book.lookup(chosen))
        return logits

    def generate(self, events, samples=1, stochastic=True):
        if self.training: raise RuntimeError('Generation requires eval mode')
        memory=self.audio_memory(events).repeat_interleave(samples,0)
        levels=[]; codes=[]; logits=[]
        for level,(n,book) in enumerate(zip(self.tokenizer.lengths,self.tokenizer.books)):
            logit=self.predictor.forward_level(memory,levels,level,n)
            c=torch.distributions.Categorical(logits=logit).sample() if stochastic else logit.argmax(-1)
            logits.append(logit); codes.append(c); levels.append(book.lookup(c))
        probs=self.tokenizer.decode_codes(codes).softmax(-1)
        return probs.reshape(len(events),samples,4,10),codes,logits

    def differentiable_generation(self, events):
        """Target-free hard argmax rollout with softmax straight-through gradients.

        Forward embeddings are discrete, as in deterministic inference. Only
        backward uses the soft categorical Jacobian. Frozen decoder parameters
        still permit gradients with respect to the generated latent input.
        """
        memory=self.audio_memory(events)
        levels=[]; codes=[]; code_logits=[]; digit_logits=[]
        for level,(n,book) in enumerate(zip(self.tokenizer.lengths,self.tokenizer.books)):
            logit=self.predictor.forward_level(memory,levels,level,n)
            probabilities=logit.softmax(-1)
            code=logit.argmax(-1)
            hard=F.one_hot(code,self.cfg.codebook_size).to(probabilities.dtype)
            weights=hard+(probabilities-probabilities.detach())
            embedding_table=book.lookup(torch.arange(self.cfg.codebook_size,device=events.device))
            levels.append(weights @ embedding_table)
            codes.append(code);code_logits.append(logit)
            reconstruction=sum(self.tokenizer.resize(q,4) for q in levels)
            digit_logits.append(self.tokenizer.decoder(reconstruction))
        return {'codes':codes,'code_logits':code_logits,'digit_logits':digit_logits}

    @torch.no_grad()
    def sequence_probabilities(self, events, samples=1):
        probs,_,_=self.generate(events,samples)
        return mixture_scores(probs)


def mixture_scores(probs):
    """Exact 10,000-sequence enumeration of equally weighted sampled hierarchies."""
    universe=torch.cartesian_prod(*[torch.arange(10,device=probs.device)]*4)
    log_scores=sum(probs[:,:,i,universe[:,i]].clamp_min(1e-30).log() for i in range(4))
    return torch.logsumexp(log_scores,dim=1).exp()/probs.shape[1]
