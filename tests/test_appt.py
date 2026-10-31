import torch
from ppf.appt import (LocalAcousticTokenizer,LocalReranker,LocalHierarchicalRefiner,
                      AdaptivePossibilityTree,uncertainty_gate,possibility_tree)
from ppf.model import CNNBaseline


def confident(digit):
    p=torch.full((10,),.02/9);p[digit]=.98;return p


def test_single_uncertain_position_has_ten_branches_and_reports_mass():
    p=torch.stack([confident(1),torch.full((10,),.1),confident(3),confident(4)])
    expand=torch.tensor([False,True,False,False])
    tree=possibility_tree(p,expand,top_k=10)
    assert tree['branches']==10
    assert set(tree['sequences'][:,1].tolist())==set(range(10))
    assert (tree['sequences'][:,[0,2,3]]==torch.tensor([1,3,4])).all()
    assert abs(tree['retained_probability_mass']-.98**3)<1e-6


def test_high_confidence_error_can_be_reopened_without_labels():
    base=torch.stack([confident(1),confident(2),confident(3),confident(4)])
    verification=base.clone();verification[1]=confident(8)
    expand,reopened=uncertainty_gate(base,verification,.9,.8)
    assert expand.tolist()==[False,True,False,False]
    assert reopened.tolist()==[False,True,False,False]
    assert possibility_tree(base,expand,10)['branches']==10


def initialized_tokenizer():
    tok=LocalAcousticTokenizer();features=torch.randn(32,64,31)
    tok.fit_projection(features);z=tok.project(features);tok(z)
    return tok.freeze(),z


def test_local_hierarchy_shapes_hard_st_equivalence_and_frozen_state():
    torch.manual_seed(3);tok,z=initialized_tokenizer()
    model=LocalHierarchicalRefiner(tok).eval()
    before={k:v.clone() for k,v in tok.state_dict().items()}
    hard=model(z);straight=model(z,differentiable=True)
    assert [c.shape[1] for c in hard['codes']]==[1,2,4]
    assert all(torch.equal(a,b) for a,b in zip(hard['codes'],straight['codes']))
    assert all(torch.allclose(a,b,atol=1e-6) for a,b in zip(hard['digit_logits'],straight['digit_logits']))
    loss=sum(v.square().mean() for v in straight['digit_logits']);loss.backward()
    assert model.code_heads[-1].weight.grad.norm()>0
    assert all(torch.equal(before[k],v) for k,v in tok.state_dict().items())


def test_erased_audio_keeps_uniform_probabilities_and_all_digits_possible():
    torch.manual_seed(4);tok,_=initialized_tokenizer()
    system=AdaptivePossibilityTree(CNNBaseline(),tok,LocalReranker(),LocalHierarchicalRefiner(tok),
                                  {'cnn':1.,'local':1.,'appt':1.},.05,.999)
    events=torch.ones(4,2400);events[0]=0
    result=system.predict(events,top_k=10)
    assert torch.equal(result['digit_probabilities'][0],torch.full((10,),.1))
    assert result['branches']==10 and result['refined_positions']==0
    assert set(result['sequences'][:,0].tolist())==set(range(10))
