import torch
from torch import nn
from ppf.config import ModelConfig
from ppf.features import LogMelSpectrogram
from ppf.safe_gate import RelativeFloorLogMel,SafeAdaptiveTree,gate_features


def test_relative_floor_zero_matches_original_and_zero_audio_is_finite():
    cfg=ModelConfig();a=LogMelSpectrogram();b=RelativeFloorLogMel(cfg,0)
    x=torch.randn(3,2400)
    assert torch.equal(a(x),b(x))
    corrected=RelativeFloorLogMel(cfg,.001)
    assert torch.isfinite(corrected(torch.zeros(2,2400))).all()


def probabilities(digit,confidence):
    p=torch.full((10,),(1-confidence)/9);p[digit]=confidence;return p


class DummyCNN(nn.Module):
    def encoder(self,events):return torch.zeros(len(events),64,31)
    def head(self,features):return probabilities(1,.98).log().repeat(len(features),1)


class DummyTokenizer:
    def freeze(self):return self
    def project(self,f):return torch.zeros(len(f),4,32)


class Proposal(nn.Module):
    def __init__(self,confidence,hierarchical=False):
        super().__init__();self.confidence=confidence;self.hierarchical=hierarchical
    def forward(self,z):
        logits=probabilities(3,self.confidence).log().repeat(len(z),1)
        return {'digit_logits':[logits]*3} if self.hierarchical else logits


class ConfidenceGate(nn.Module):
    def score(self,f):return f[:,3]


def system(candidate_confidence):
    return SafeAdaptiveTree(DummyCNN(),DummyTokenizer(),Proposal(.99),Proposal(candidate_confidence,True),
                            {'cnn':1.,'local':1.,'appt':1.},ConfidenceGate(),.9,.9,.9)


def test_route_does_not_automatically_accept_a_weak_proposal():
    result=system(.7).predict(torch.ones(4,2400),'appt_gated')
    assert result['expanded'].all() and not result['accepted'].any()
    assert (result['digit_probabilities'].argmax(-1)==1).all()


def test_same_gate_accepts_strong_proposals_and_can_reopen_confident_base():
    s=system(.99)
    hierarchical=s.predict(torch.ones(4,2400),'appt_gated')
    conventional=s.predict(torch.ones(4,2400),'local_gated')
    assert hierarchical['accepted'].all()
    assert (hierarchical['digit_probabilities'].argmax(-1)==3).all()
    assert torch.equal(hierarchical['accepted'],conventional['accepted'])
    assert torch.allclose(hierarchical['digit_probabilities'],conventional['digit_probabilities'])


def test_erasure_overrides_gate_and_keeps_all_digits_possible():
    result=system(.7).predict(torch.zeros(4,2400),'appt_gated')
    assert result['branches']==10000 and result['refined_positions']==0
    assert torch.equal(result['digit_probabilities'],torch.full((4,10),.1))
