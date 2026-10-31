"""Controlled local feature alternatives; no gate or CNN changes."""
import torch
from torch import nn
from .appt import LocalReranker,LocalHierarchicalRefiner,resize
from .sequence import EMACodebook


class HierarchyStudyVariant(nn.Module):
    def __init__(self,kind,tokenizer,flat_book,head_state):
        super().__init__();self.kind=kind;self.tokenizer=tokenizer.freeze();self.flat_book=flat_book.eval()
        if kind=='D':
            self.model=LocalHierarchicalRefiner(tokenizer)
            self.model.ranker.load_state_dict(head_state)
        else:
            self.model=LocalReranker();self.model.load_state_dict(head_state)

    def train(self,mode=True):
        super().train(mode);self.tokenizer.eval();self.flat_book.eval();return self

    @torch.no_grad()
    def representation(self,z):
        if self.kind=='A':return z
        if self.kind=='B':return self.flat_book(z)[0]
        if self.kind=='C':return self.tokenizer(z)['prefixes'][-1]
        out=self.model(z)
        return sum(resize(book.lookup(c),4) for book,c in zip(self.tokenizer.books,out['codes']))

    def forward(self,z,differentiable=False):
        if self.kind=='D':return self.model(z,differentiable=differentiable)['digit_logits'][-1]
        return self.model(self.representation(z))

    def loss(self,z,labels,codes):
        import torch.nn.functional as F
        if self.kind!='D':return F.cross_entropy(self(z),labels)
        out=self.model(z,differentiable=True)
        return sum(F.cross_entropy(l,labels) for l in out['digit_logits'])/3+.1*sum(
            F.cross_entropy(l.flatten(0,1),c.flatten()) for l,c in zip(out['code_logits'],codes))/3
