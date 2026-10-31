import copy
import torch
from ppf.appt import LocalAcousticTokenizer,LocalReranker,possibility_tree
from ppf.sequence import EMACodebook
from ppf.hierarchy_study import HierarchyStudyVariant


def fixtures():
    torch.manual_seed(27);tok=LocalAcousticTokenizer();features=torch.randn(32,64,31)
    tok.fit_projection(features);z=tok.project(features);tok(z);tok.freeze()
    flat=EMACodebook(64,32);flat(z);flat.eval()
    return tok,flat,z,LocalReranker().state_dict()


def test_fine_only_is_independent_of_coarse_residual_books():
    tok,flat,z,head=fixtures();m=HierarchyStudyVariant('B',tok,flat,head).eval()
    before=m.representation(z)
    with torch.no_grad():tok.books[0].weight.add_(100)
    assert torch.equal(before,m.representation(z))


def test_observed_hierarchy_matches_tokenizer_and_all_models_freeze_books():
    tok,flat,z,head=fixtures();m=HierarchyStudyVariant('C',tok,flat,head).train()
    before={k:v.clone() for k,v in tok.state_dict().items()}
    expected=tok(z)['prefixes'][-1]
    assert torch.equal(m.representation(z),expected)
    loss=m.loss(z,torch.randint(10,(32,)),tok(z)['codes']);loss.backward()
    assert all(torch.equal(before[k],v) for k,v in tok.state_dict().items())
    full=HierarchyStudyVariant('D',copy.deepcopy(tok),copy.deepcopy(flat),head).eval()
    assert full(z).shape==(32,10)
    assert torch.equal(full.model.ranker.head[0].weight,head['head.0.weight'])


def test_reranking_preserves_frozen_candidate_support():
    torch.manual_seed(2)
    base=torch.randn(4,10).softmax(-1);route=torch.tensor([False,True,False,True])
    changed=base.clone();changed[route]=torch.randn(2,10).softmax(-1)
    a=possibility_tree(base,route,100);b=possibility_tree(changed,route,100)
    assert a['branches']==b['branches']==100
    assert set(map(tuple,a['sequences'].tolist()))==set(map(tuple,b['sequences'].tolist()))
