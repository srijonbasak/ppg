import torch
import pytest
from ppf.config import ModelConfig
from ppf.sequence import CleanLabelTokenizer, SequencePPF, mixture_scores


def cfg():
    return ModelConfig(model_dim=32,code_dim=32,codebook_size=16,transformer_layers=1)


def test_fixed_scales_nested_dropout_and_gradients():
    torch.manual_seed(4)
    tok=CleanLabelTokenizer(cfg()).train()
    y=torch.randint(10,(16,4))
    out=tok(y,prefix=1)
    assert [c.shape[1] for c in out['codes']]==[1,2,4]
    assert out['prefix']==1
    out['loss'].backward()
    assert torch.isfinite(tok.embedding.weight.grad).all()
    assert tok.embedding.weight.grad.abs().sum()>0
    assert all(not book.weight.requires_grad for book in tok.books)
    tok.eval()
    out=tok(y)
    assert torch.allclose(tok.decode_codes(out['codes']),out['logits'][-1],atol=1e-5)
    # A decoded coarse prefix must not depend on finer codes.
    coarse=tok.decoder(tok.resize(tok.books[0].lookup(out['codes'][0]),4))
    assert torch.allclose(coarse,out['logits'][0],atol=1e-5)


def test_frozen_tokenizer_and_target_free_generation(monkeypatch):
    tok=CleanLabelTokenizer(cfg())
    tok(torch.randint(10,(16,4)))
    model=SequencePPF(tok,cfg()).train()
    assert not tok.training and all(not p.requires_grad for p in tok.parameters())
    before={k:v.clone() for k,v in tok.state_dict().items()}
    def forbidden(*args,**kwargs):raise AssertionError('Target tokenizer accessed during generation')
    monkeypatch.setattr(tok,'forward',forbidden)
    model.eval()
    probs,codes,_=model.generate(torch.randn(2,4,2400),samples=3)
    assert probs.shape==(2,3,4,10)
    assert [c.shape for c in codes]==[torch.Size([6,1]),torch.Size([6,2]),torch.Size([6,4])]
    assert all(torch.equal(before[k],v) for k,v in tok.state_dict().items())


def test_generated_prior_is_carried_forward(monkeypatch):
    model=SequencePPF(CleanLabelTokenizer(cfg()),cfg()).eval()
    seen=[]
    def level(memory,previous,index,length):
        seen.append([p.clone() for p in previous])
        logits=torch.full((len(memory),length,16),-100.)
        logits[:,:,index+2]=100.
        return logits
    monkeypatch.setattr(model.predictor,'forward_level',level)
    _,codes,_=model.generate(torch.zeros(1,4,2400),stochastic=False)
    assert all(torch.equal(seen[2][i],model.tokenizer.books[i].lookup(codes[i])) for i in range(2))


def test_mixture_preserves_joint_mass_and_duplicate_hypotheses():
    p=torch.zeros(1,3,4,10)
    p[0,0,:,1]=1;p[0,1,:,2]=1;p[0,2,:,1]=1
    scores=mixture_scores(p)
    assert torch.allclose(scores.sum(-1),torch.ones(1))
    assert torch.isclose(scores[0,1111],torch.tensor(2/3))
    assert torch.isclose(scores[0,2222],torch.tensor(1/3))
    assert scores[0,1122]<1e-20


def test_teacher_and_generated_prefix_training_match_explicit_paths():
    torch.manual_seed(22)
    tok=CleanLabelTokenizer(cfg())
    y=torch.randint(10,(4,4));tok(y)
    model=SequencePPF(tok,cfg()).eval()
    x=torch.randn(4,4,2400)
    codes=tok(y)['codes']
    teacher=model.teacher_logits(x,codes)
    levels=[book.lookup(c) for book,c in zip(tok.books,codes)]
    manual=model.predictor(model.audio_memory(x),levels)
    assert all(torch.equal(a,b) for a,b in zip(teacher,manual))
    mixed=model.training_logits(x,codes,generated_prefix_probability=1)
    _,_,generated=model.generate(x,stochastic=False)
    assert all(torch.equal(a,b) for a,b in zip(mixed,generated))


def test_oracle_codes_use_same_inference_batch_and_decoder_mapping(monkeypatch):
    torch.manual_seed(7)
    tok=CleanLabelTokenizer(cfg())
    labels=torch.randint(10,(8,4));tok(labels)
    model=SequencePPF(tok,cfg()).eval()
    codes=tok(labels)['codes']
    def forced(memory,previous,level,length):
        expected=codes[level].repeat_interleave(3,0)
        for l,p in enumerate(previous):
            assert torch.equal(p,tok.books[l].lookup(codes[l].repeat_interleave(3,0)))
        logits=torch.full((*expected.shape,16),-100.)
        return logits.scatter_(-1,expected[:,:,None],100.)
    monkeypatch.setattr(model.predictor,'forward_level',forced)
    probs,_,_=model.generate(torch.randn(8,4,2400),samples=3,stochastic=False)
    expected=tok.decode_codes(codes).softmax(-1)
    assert torch.equal(probs,expected[:,None].expand(-1,3,-1,-1))


def test_first_level_sampling_matches_conditional_distribution(monkeypatch):
    torch.manual_seed(18)
    model=SequencePPF(CleanLabelTokenizer(cfg()),cfg()).eval()
    def fixed_distribution(memory,previous,level,length):
        logits=torch.full((len(memory),length,16),-100.)
        logits[:,:,0]=0
        logits[:,:,1]=torch.log(torch.tensor(3.))
        return logits
    monkeypatch.setattr(model.predictor,'forward_level',fixed_distribution)
    _,codes,_=model.generate(torch.zeros(1,4,2400),samples=512,stochastic=True)
    proportion=(codes[0]==1).float().mean()
    assert abs(float(proportion)-.75)<.07
    assert codes[0].unique().tolist()==[0,1]
