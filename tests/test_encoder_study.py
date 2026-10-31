import torch
import torch.nn.functional as F
from ppf.model import CNNBaseline
from ppf.appt import LocalAcousticTokenizer
from ppf.graph_ncm import AcousticNCMTransformer
from ppf.encoder_study import encoder_taps,ChannelProjection,ConditionedNCM,missing_frequency,digit_input,DigitProbe


def test_taps_reproduce_encoder_without_temporal_downsampling():
    cnn=CNNBaseline().eval();x=torch.randn(2,2400)
    taps=encoder_taps(cnn,x)
    assert taps['mel'].shape==(2,40,31)
    assert all(taps[k].shape==(2,64,31) for k in ['stem','block1','block2','final'])
    assert torch.equal(taps['final'],cnn.encoder(x))
    assert list(encoder_taps(cnn,x,'stem'))==['mel','stem']


def test_dense_memory_contains_the_four_token_condition():
    torch.manual_seed(6);x=torch.randn(32,64,31);tok=LocalAcousticTokenizer();tok.fit_projection(x)
    adapter=ChannelProjection(64)
    for key in ['mean','std','projection','scale']:getattr(adapter,key).copy_(getattr(tok,key))
    dense=adapter.frames(x);coarse=adapter(x)
    assert dense.shape==(32,31,32) and coarse.shape==(32,4,32)
    assert torch.allclose(coarse,tok.project(x),atol=2e-6)
    assert torch.allclose(coarse,F.adaptive_avg_pool1d(dense.transpose(1,2),4).transpose(1,2),atol=1e-7)


def test_variable_memory_changes_no_generator_parameters_or_code_shapes():
    torch.manual_seed(7);tok=LocalAcousticTokenizer();tok(torch.randn(8,4,32));tok.freeze()
    old=AcousticNCMTransformer(tok).eval();model=ConditionedNCM(tok).eval();model.load_state_dict(old.state_dict())
    assert sum(p.numel() for p in old.parameters())==sum(p.numel() for p in model.parameters())==34570
    z=torch.randn(3,4,32)
    assert torch.equal(old(z)['digit_logits'][-1],model(z)['digit_logits'][-1])
    x=torch.randn(3,31,32,requires_grad=True);out=model(x,differentiable=True)
    assert [v.shape[1] for v in out['codes']]==[1,2,4]
    out['digit_logits'][-1].square().mean().backward()
    assert x.grad.norm()>0 and model.audio_position.grad.norm()>0


def test_all_native_stage_probes_have_the_same_input_and_parameter_budget():
    cnn=CNNBaseline().eval();taps=encoder_taps(cnn,torch.randn(2,2400))
    for name in ['mel','stem','block1','block2','final','final_mean']:
        x=digit_input(taps,name,{})
        assert x.shape==(2,256)
        assert sum(p.numel() for p in DigitProbe(torch.zeros(256),torch.ones(256)).parameters())==17098


def test_missing_column_identity_is_not_encoded_in_waveforms():
    first=torch.tensor([[1,4,7,0]])
    second=torch.tensor([[3,6,9,0]])
    assert torch.equal(missing_frequency(first,93),missing_frequency(second,93))
