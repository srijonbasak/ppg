import torch
from ppf.config import ModelConfig
from ppf.compositional import CompositionalTokenizer
from ppf.sequence import SequencePPF


def config():
    return ModelConfig(model_dim=32,code_dim=32,codebook_size=16,transformer_layers=1)


def test_subspace_residual_hierarchy_and_local_code_invariance():
    torch.manual_seed(31)
    tok=CompositionalTokenizer(config())
    labels=torch.tensor([[0,1,2,3],[4,5,6,7],[8,9,0,1],[2,3,4,5]])
    tok(labels);tok.eval()
    out=tok(labels)
    assert [v.shape[1] for v in out['codes']]==[1,2,4]
    for q,book in zip(out['levels'],tok.books):
        assert torch.count_nonzero(q*(1-book.mask))==0
    changed=labels.clone();changed[:,1]=(changed[:,1]+1)%10
    altered=tok(changed)
    assert torch.equal(out['codes'][2][:,[0,2,3]],altered['codes'][2][:,[0,2,3]])
    assert (out['codes'][2][:,1]!=altered['codes'][2][:,1]).all()
    assert torch.equal(out['codes'][1][:,1],altered['codes'][1][:,1])
    assert torch.allclose(tok.decode_codes(out['codes']),out['logits'][-1],atol=1e-6)


def test_structured_tokenizer_freezes_and_uses_existing_generator():
    tok=CompositionalTokenizer(config())
    tok(torch.randint(10,(16,4)))
    model=SequencePPF(tok,config()).eval()
    before={k:v.clone() for k,v in tok.state_dict().items()}
    probs,codes,_=model.generate(torch.zeros(2,4,2400),stochastic=False)
    assert probs.shape==(2,1,4,10)
    assert all(torch.equal(before[k],v) for k,v in tok.state_dict().items())
    assert not any(p.requires_grad for p in tok.parameters())


def test_straight_through_rollout_matches_hard_inference_without_targets(monkeypatch):
    torch.manual_seed(9)
    tok=CompositionalTokenizer(config());tok(torch.randint(10,(16,4)))
    model=SequencePPF(tok,config()).eval()
    def forbidden(*args,**kwargs):raise AssertionError('Target labels/codes accessed')
    monkeypatch.setattr(tok,'forward',forbidden)
    x=torch.randn(3,4,2400)
    output=model.differentiable_generation(x)
    probs,codes,logits=model.generate(x,stochastic=False)
    assert all(torch.equal(a,b) for a,b in zip(output['codes'],codes))
    assert all(torch.equal(a,b) for a,b in zip(output['code_logits'],logits))
    assert torch.allclose(output['digit_logits'][-1].softmax(-1),probs[:,0],atol=1e-7)
    levels=[]
    for l,(book,c) in enumerate(zip(tok.books,codes)):
        levels.append(book.lookup(c))
        expected=tok.decoder(sum(tok.resize(q,4) for q in levels))
        assert torch.allclose(output['digit_logits'][l],expected,atol=1e-7)


def test_auxiliary_gradients_reach_each_code_level_with_frozen_decoder():
    import torch.nn.functional as F
    torch.manual_seed(15)
    tok=CompositionalTokenizer(config());tok(torch.randint(10,(16,4)))
    model=SequencePPF(tok,config()).train()
    before={k:v.clone() for k,v in tok.state_dict().items()}
    output=model.differentiable_generation(torch.randn(4,4,2400))
    labels=torch.randint(10,(4,4));loss=0
    for l,logits in enumerate(output['digit_logits']):
        term=F.cross_entropy(logits.flatten(0,1),labels.flatten())
        gradient=torch.autograd.grad(term,output['code_logits'][l],retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and gradient.norm()>0
        loss=loss+term
    loss.backward()
    assert model.predictor.heads[-1].weight.grad.norm()>0
    assert all(p.grad is None for p in tok.parameters())
    assert all(torch.equal(before[k],v) for k,v in tok.state_dict().items())
