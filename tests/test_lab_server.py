import inspect
import pytest
import torch
from ppf.lab_server import make_audio,predict


def test_audio_controls_are_repeatable_and_local_to_selected_event():
    clean=make_audio('7283','clean',.5,3,42)
    for condition in ['noise','partial_erasure','missing_frequency','interference']:
        a=make_audio('7283',condition,.8,3,42)
        assert a.shape==(4,2400) and torch.isfinite(a).all() and a.abs().max()<=1
        assert torch.equal(a,make_audio('7283',condition,.8,3,42))
        assert torch.equal(a[:3],clean[:3]) and not torch.equal(a[3],clean[3])
    assert list(inspect.signature(predict).parameters)==['events','seed','variant']


def test_invalid_audio_requests_are_rejected():
    for sequence in ['123','12345','abcd','１２３４']:
        with pytest.raises(ValueError):make_audio(sequence,'clean',.5,3,42)
    with pytest.raises(ValueError):make_audio('1234','unknown',.5,3,42)
    with pytest.raises(ValueError):make_audio('1234','noise',2,3,42)


def test_real_checkpoints_return_consistent_ranking_and_code_shapes():
    torch.set_num_threads(1);a=make_audio('7283','clean',.5,3,42)
    for variant in ['original_joint','original_aligned','recognition']:
        r=predict(a,317,variant)
        assert [len(level[0]) for level in r['codes']]==[1,2,4]
        assert all(len(row)==10 for row in r['final'])
        for i,accepted in enumerate(r['accepted']):
            if not accepted:assert r['final'][i]==r['base'][i]
        assert r['candidate_count']==10**sum(r['route'])
        for rows in r['ranking'].values():
            assert 1<=len(rows)<=5
            assert all(a['probability']>=b['probability'] for a,b in zip(rows,rows[1:]))
