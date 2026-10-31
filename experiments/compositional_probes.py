"""Frozen-representation linear probes and one-digit intervention tests."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ppf.config import ModelConfig
from ppf.model import CNNBaseline
from ppf.sequence import CleanLabelTokenizer

ROOT=Path('compositional_results');C=json.loads(Path('experiments/compositional_config.json').read_text());CFG=ModelConfig()

def original_tokenizer():
    tok=CleanLabelTokenizer(CFG)
    tok.load_state_dict(torch.load('representation_results/tokenizer.pt',weights_only=True)['model'])
    return tok.freeze()

def load_data():
    raw=torch.load('representation_results/audio_data.pt',weights_only=False)
    seen=set(map(tuple,raw['train'][1].tolist()))
    selection={'train':list(range(len(raw['train'][1])))}
    selection['val']=[i for i,y in enumerate(raw['val'][1].tolist()) if tuple(y) not in seen]
    seen.update(map(tuple,raw['val'][1].tolist()))
    selection['test']=[i for i,y in enumerate(raw['test'][1].tolist()) if tuple(y) not in seen]
    data={s:(x[selection[s]],y[selection[s]],[cond[i] for i in selection[s]]) for s,(x,y,cond) in raw.items()}
    pairs={f'{a},{b}':set((v[a],v[b]) for v in raw['train'][1].tolist()) for a in range(4) for b in range(a+1,4)}
    assert all(len(v)==100 for v in pairs.values())
    for split in ['val','test']:
        assert not set(map(tuple,data[split][1].tolist()))&set(map(tuple,data['train'][1].tolist()))
    (ROOT/'split_indices.json').write_text(json.dumps({'indices':selection,'counts':{s:len(d[1]) for s,d in data.items()},
        'distinct_label_sequences':{s:len(set(map(tuple,d[1].tolist()))) for s,d in data.items()},
        'training_position_pair_coverage':{k:len(v) for k,v in pairs.items()}},indent=2))
    return data

@torch.inference_mode()
def representations(tok,labels):
    out=tok(labels);acc=torch.zeros(len(labels),4,64);prefix=[]
    for q in out['levels']:
        acc=acc+tok.resize(q,4);prefix.append(acc.flatten(1).clone())
    return prefix,out['codes']


def train_probe(train_x,train_y,test_x,test_y,classes,positions):
    torch.manual_seed(C['seed'])
    mean=train_x.mean(0);std=train_x.std(0).clamp_min(.01)
    tx=(train_x-mean)/std;vx=(test_x-mean)/std
    model=torch.nn.Linear(tx.shape[-1],positions*classes)
    opt=torch.optim.Adam(model.parameters(),lr=C['probe_learning_rate'])
    for _ in range(C['probe_steps']):
        opt.zero_grad(set_to_none=True);logits=model(tx).reshape(-1,positions,classes)
        loss=F.cross_entropy(logits.flatten(0,1),train_y.flatten());loss.backward();opt.step()
    with torch.inference_mode():
        train_ok=model(tx).reshape(-1,positions,classes).argmax(-1)==train_y
        ok=model(vx).reshape(-1,positions,classes).argmax(-1)==test_y
    return {'train_accuracy_by_position':train_ok.float().mean(0).tolist(),
            'test_accuracy_by_position':ok.float().mean(0).tolist(),
            'test_all_positions_correct':float(ok.all(-1).float().mean()),
            'parameters':sum(p.numel() for p in model.parameters())}

@torch.inference_mode()
def neighborhood(tok):
    rng=np.random.default_rng(C['seed'])
    anchor=torch.tensor(rng.integers(0,10,(C['neighbor_examples_per_position'],4)))
    base=tok(anchor);result={}
    for pos in range(4):
        altered=anchor.clone();delta=torch.tensor(rng.integers(1,10,len(anchor)))
        altered[:,pos]=(altered[:,pos]+delta)%10
        other=tok(altered);rows=[]
        for l in range(3):
            a=base['codes'][l];b=other['codes'][l]
            qa=tok.books[l].lookup(a);qb=tok.books[l].lookup(b)
            rows.append({'changed_code_fraction_by_slot':(a!=b).float().mean(0).tolist(),
                         'any_code_changed':float((a!=b).any(-1).float().mean()),
                         'mean_squared_embedding_distance':float((qa-qb).square().mean())})
        result[f'change_position_{pos+1}']=rows
    unrelated=anchor.clone()
    # All four positions differ: an explicitly defined unrelated control.
    unrelated=(unrelated+torch.tensor(rng.integers(1,10,tuple(anchor.shape))))%10
    other=tok(unrelated)
    result['unrelated_all_four_different']=[{'changed_code_fraction_by_slot':(a!=b).float().mean(0).tolist(),
        'any_code_changed':float((a!=b).any(-1).float().mean()),
        'mean_squared_embedding_distance':float((tok.books[l].lookup(a)-tok.books[l].lookup(b)).square().mean())}
        for l,(a,b) in enumerate(zip(base['codes'],other['codes']))]
    return result


def run_probes(tok,name,data,audio_features):
    train=data['train'][1];test=data['test'][1]
    with torch.inference_mode():
        a,ac=representations(tok,train);b,bc=representations(tok,test)
    result={'tokenizer_digit_probes':[], 'audio_to_code_probes':[],'neighbors':neighborhood(tok)}
    for l in range(3):
        result['tokenizer_digit_probes'].append(train_probe(a[l].clone(),train,b[l].clone(),test,10,4))
        result['audio_to_code_probes'].append(train_probe(audio_features['train'],ac[l].clone(),audio_features['test'],bc[l].clone(),64,ac[l].shape[1]))
    (ROOT/f'{name}_probes.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.manual_seed(C['seed']);torch.use_deterministic_algorithms(True)
    (ROOT/'config.json').write_text(json.dumps(C,indent=2));data=load_data()
    cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.eval()
    features={}
    with torch.inference_mode():
        for split,(x,y,cond) in data.items():
            chunks=[cnn.encoder(z.flatten(0,1)).mean(-1).reshape(len(z),256) for z in x.split(64)]
            features[split]=torch.cat(chunks).clone()
    # Explicitly leave inference mode before training probe weights.
    features={k:v.clone() for k,v in features.items()}
    torch.save(features,ROOT/'frozen_audio_features.pt')
    cnnprobe=train_probe(features['train'],data['train'][1],features['test'],data['test'][1],10,4)
    (ROOT/'cnn_digit_probe.json').write_text(json.dumps(cnnprobe,indent=2))
    result=run_probes(original_tokenizer(),'original',data,features)
    print(json.dumps({'cnn_digit_probe':cnnprobe,'original_token_digit_probe':result['tokenizer_digit_probes'],
                      'original_audio_code_probe':result['audio_to_code_probes']}),flush=True)

if __name__=='__main__':main()
