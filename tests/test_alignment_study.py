import torch
from ppf.appt import LocalAcousticTokenizer
from ppf.encoder_study import ConditionedNCM
from ppf.alignment_study import LosslessEvidenceAdapter,coarse_summary,level1_path,oracle_rollout,aligned_loss


def fixture():
    torch.manual_seed(19)
    tok=LocalAcousticTokenizer();tok(torch.randn(12,4,32));tok.freeze()
    return ConditionedNCM(tok).eval()


def test_reversible_adapter_preserves_anchors_and_native_evidence():
    a=LosslessEvidenceAdapter();stem=torch.randn(8,64,31);final=torch.randn_like(stem);anchors=torch.randn(8,4,32)
    a.fit(stem,final);memory=a(stem,final,anchors);x,y=a.unpack(memory)
    assert memory.shape==(8,128,32) and sum(p.numel() for p in a.parameters())==0
    assert torch.equal(memory[:,:4],anchors)
    assert torch.allclose(x,stem,atol=1e-6) and torch.allclose(y,final,atol=1e-6)
    assert torch.allclose(coarse_summary(x,y),coarse_summary(stem,final),atol=1e-6)


def test_actual_level1_trace_reproduces_prediction_for_both_memory_lengths():
    model=fixture()
    for n in [4,128]:
        z=torch.randn(3,n,32)
        assert torch.equal(level1_path(model,z)['logits'],model(z)['code_logits'][0])


def test_oracle_prefix_is_explicit_and_remaining_codes_are_generated():
    model=fixture();z=torch.randn(3,4,32);prefix=torch.tensor([[1],[2],[3]])
    out=oracle_rollout(model,z,prefix)
    assert torch.equal(out['codes'][0],prefix)
    assert [c.shape[1] for c in out['codes']]==[1,2,4]
    for i in [1,2]:assert torch.equal(out['codes'][i],out['code_logits'][i].argmax(-1))


def test_alignment_trains_predictor_and_decoder_without_unfreezing_codes():
    model=fixture();z=torch.randn(5,4,32);target=model.tokenizer(z)['codes'];labels=torch.randint(0,10,(5,))
    loss=aligned_loss(model,z,labels,target);loss.backward()
    assert torch.isfinite(loss)
    assert model.predictor.heads[0].weight.grad.norm()>0
    assert next(model.ranker.parameters()).grad.norm()>0
    assert all(p.grad is None for p in model.tokenizer.parameters())


def test_digit_labels_cannot_change_aligned_predictor_gradients():
    model=fixture();z=torch.randn(5,4,32);target=model.tokenizer(z)['codes']
    aligned_loss(model,z,torch.zeros(5,dtype=torch.long),target).backward()
    predictor=model.predictor.heads[0].weight.grad.clone();decoder=next(model.ranker.parameters()).grad.clone()
    model.zero_grad(set_to_none=True)
    aligned_loss(model,z,torch.ones(5,dtype=torch.long),target).backward()
    assert torch.equal(predictor,model.predictor.heads[0].weight.grad)
    assert not torch.equal(decoder,next(model.ranker.parameters()).grad)
