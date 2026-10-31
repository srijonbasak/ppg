import torch
from ppf.graph_ncm import frequency_evidence,graph_messages,EvidenceProjection,AcousticNCMTransformer
from ppf.appt import LocalAcousticTokenizer


def test_physical_graph_supports_two_frequencies_but_not_one():
    e=torch.full((2,4,7),-12.)
    e[:,:,0]=-1.;e[:,:,5]=-1. # 697 + 1336 Hz = digit 2
    assert (graph_messages(e).argmax(-1)==2).all()
    e[:,:,5]=-12.
    support=graph_messages(e)
    assert torch.equal(support,support[:,:,:1].expand_as(support))


def test_graph_projection_is_label_free_and_zero_evidence_is_finite():
    e=frequency_evidence(torch.zeros(3,2400));assert torch.isfinite(e).all()
    assert torch.equal(e,e[:,:,:1].expand_as(e))
    enc=EvidenceProjection(True);z=torch.randn(3,4,32);enc.fit(z,e)
    assert enc(z,e).shape==(3,4,32) and torch.isfinite(enc(z,e)).all()
    assert [sum(f in pair for pair in enc.pairs.tolist()) for f in range(7)]==[3,3,3,1,3,4,3]


def test_transformer_uses_generated_levels_and_preserves_frozen_codes(monkeypatch):
    torch.manual_seed(51);tok=LocalAcousticTokenizer();z=torch.randn(4,4,32);tok(z);tok.freeze()
    model=AcousticNCMTransformer(tok).eval();before={k:v.clone() for k,v in tok.state_dict().items()}
    seen=[]
    original=model.predictor.forward_level
    def observed(audio,previous,level,length):
        seen.append((level,len(previous),length));return original(audio,previous,level,length)
    monkeypatch.setattr(model.predictor,'forward_level',observed)
    hard=model(z);assert seen==[(0,0,1),(1,1,2),(2,2,4)]
    soft=model(z,differentiable=True)
    assert all(torch.equal(a,b) for a,b in zip(hard['codes'],soft['codes']))
    assert all(torch.allclose(a,b,atol=1e-6) for a,b in zip(hard['digit_logits'],soft['digit_logits']))
    soft['digit_logits'][-1].square().mean().backward()
    assert model.predictor.heads[-1].weight.grad.norm()>0
    assert all(torch.equal(before[k],v) for k,v in tok.state_dict().items())


def test_clear_missing_frequency_band_retains_compatible_uncertainty():
    import math
    from ppf.graph_ncm import missing_band_distribution
    t=torch.arange(1440)/8000
    rows=[]
    for frequency in [697,770,852]:
        tone=.36*torch.sin(2*math.pi*frequency*t+1.)
        ramp=torch.linspace(0,1,40);tone[:40]*=ramp;tone[-40:]*=ramp.flip(0)
        rows.append(torch.nn.functional.pad(tone,(0,960)))
    mask,p=missing_band_distribution(torch.stack(rows))
    assert mask.all()
    assert torch.all(p.max(-1).values<=1/3)
    assert torch.allclose(p.sum(-1),torch.ones(3))
    assert (p>0).all()


def test_two_valid_frequency_components_do_not_trigger_missing_band_rule():
    import math
    from ppf.graph_ncm import missing_band_distribution
    t=torch.arange(1440)/8000
    tone=.36*(torch.sin(2*math.pi*697*t)+torch.sin(2*math.pi*1336*t+1.))
    events=torch.nn.functional.pad(tone,(0,960))[None]
    mask,_=missing_band_distribution(events)
    assert not mask.any()
