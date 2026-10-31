import torch
from ppf.appt import LocalAcousticTokenizer
from ppf.encoder_study import ConditionedNCM
from ppf.decoder_audit import intervene,recognition_loss


def make_model():
    torch.manual_seed(73);tok=LocalAcousticTokenizer();tok(torch.randn(15,4,32));tok.freeze()
    return ConditionedNCM(tok).eval()


def test_no_intervention_exactly_matches_target_free_generator():
    model=make_model();z=torch.randn(4,4,32);a=intervene(model,z);b=model(z)
    assert torch.equal(a['logits'],b['digit_logits'][-1])
    assert all(torch.equal(x,y) for x,y in zip(a['codes'],b['codes']))


def test_single_oracle_level_uses_no_other_target_codes():
    model=make_model();z=torch.randn(4,4,32);target=model.tokenizer(z)['codes'];changed=[(c+1)%64 for c in target]
    changed[1]=target[1];a=intervene(model,z,target,[1]);b=intervene(model,z,changed,[1])
    assert all(torch.equal(x,y) for x,y in zip(a['codes'],b['codes']))
    assert torch.equal(a['codes'][1],target[1])
    assert torch.equal(a['codes'][0],model(z)['codes'][0])


def test_fixed_suffix_changes_only_selected_decoder_level():
    model=make_model();z=torch.randn(4,4,32);normal=intervene(model,z);target=[(c+1)%64 for c in normal['codes']]
    out=intervene(model,z,target,[0],fixed=normal['codes'])
    assert torch.equal(out['codes'][0],target[0])
    assert all(torch.equal(out['codes'][i],normal['codes'][i]) for i in [1,2])


def test_recognition_loss_reaches_generated_code_heads_and_keeps_books_frozen():
    model=make_model();z=torch.randn(5,4,32);target=model.tokenizer(z)['codes'];y=torch.arange(5)
    loss=recognition_loss(model,z,y,target,torch.tensor([True,False,True,False,True]));loss.backward()
    assert torch.isfinite(loss)
    assert all(head.weight.grad.norm()>0 for head in model.predictor.heads)
    assert all(p.grad is None for p in model.tokenizer.parameters())
    model.zero_grad(set_to_none=True)
    assert torch.isfinite(recognition_loss(model,z,y,target,torch.zeros(5,dtype=torch.bool)))
