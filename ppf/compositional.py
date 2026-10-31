"""Controlled compositional subspace variant of the 1/2/4 residual tokenizer.

Global, pair and digit features occupy disjoint channel blocks. Quantization
still subtracts the earlier reconstruction, but restricts each codebook to its
block. This changes the representation prior, not generator size. It is a
structured residual-VQ control, not an exact reproduction of dense NCM VQ.
"""
import torch
import torch.nn.functional as F
from .sequence import CleanLabelTokenizer,EMACodebook


class SubspaceCodebook(EMACodebook):
    def __init__(self,size,dim,start,end):
        super().__init__(size,dim)
        mask=torch.zeros(dim);mask[start:end]=1
        self.register_buffer('mask',mask)

    def lookup(self,codes):
        return super().lookup(codes)*self.mask

    def forward(self,residual):
        return super().forward(residual*self.mask)


class CompositionalTokenizer(CleanLabelTokenizer):
    def __init__(self,cfg=None):
        super().__init__(cfg)
        dim=self.cfg.code_dim
        if dim%4:raise ValueError('code_dim must be divisible by four')
        width=dim//4
        self.blocks=((0,width),(width,2*width),(2*width,dim))
        self.books=torch.nn.ModuleList([SubspaceCodebook(self.cfg.codebook_size,dim,a,b) for a,b in self.blocks])
        # Target encoder is shared across digits, with no input position offset.
        # Decoder positions remain available to all representation prefixes.
        self.positions.requires_grad_(False)

    def target_features(self,labels):
        z=self.encoder(self.embedding(labels));w=self.cfg.code_dim//4
        global_context=z[:,:,:w].mean(1,keepdim=True).expand(-1,4,-1)
        pair_context=z[:,:,w:2*w].reshape(-1,2,2,w).mean(2).repeat_interleave(2,1)
        return torch.cat([global_context,pair_context,z[:,:,2*w:]],-1)

    def forward(self,labels,prefix=None):
        target=self.target_features(labels);full=torch.zeros_like(target)
        levels=[];codes=[];prefixes=[];errors=[]
        for n,book in zip(self.lengths,self.books):
            residual=self.resize(target,n)-self.resize(full.detach(),n)
            q,idx,error=book(residual)
            levels.append(q);codes.append(idx);errors.append(error)
            full=full+self.resize(q,4)
            prefixes.append(target+(full-target).detach())
        if prefix is None:prefix=int(torch.randint(1,4,()).item()) if self.training else 3
        logits=[self.decoder(p) for p in prefixes]
        loss=F.cross_entropy(logits[-1].reshape(-1,10),labels.flatten())
        loss=loss+F.cross_entropy(logits[prefix-1].reshape(-1,10),labels.flatten())+.25*torch.stack(errors).mean()
        return {'codes':codes,'levels':levels,'logits':logits,'loss':loss,
                'quantization_error':torch.stack(errors),'prefix':prefix}
