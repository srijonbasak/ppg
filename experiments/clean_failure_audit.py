"""Validation-only single-factor tests of the ideal-clean failure."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from ppf.model import CNNBaseline
from ppf.data import synthesize_sequence,SynthesisConfig
from compositional_probes import load_data,CFG

ROOT=Path('gate_results')

def clean_waveforms(labels,seed):
    rng=np.random.default_rng(seed);result=[]
    for y in labels:
        wav,meta=synthesize_sequence(''.join(map(str,y.tolist())),rng,SynthesisConfig(tone_ms=(180,180),gap_ms=(30,30)),corruption=False)
        result.append(np.stack([np.pad(wav[e['start_sample']:e['end_sample']],(0,960)) for e in meta['events']]))
    return torch.tensor(np.stack(result))


def floor_features(frontend,waveform,gamma):
    spec=torch.stft(waveform,frontend.n_fft,frontend.hop_length,window=frontend.window,return_complex=True).abs().square()
    mel=torch.einsum('mf,bft->bmt',frontend.mel,spec)
    feat=(mel+gamma*mel.mean((-2,-1),keepdim=True)).clamp_min(1e-6).log()
    return (feat-feat.mean((-2,-1),keepdim=True))/feat.std((-2,-1),keepdim=True).clamp_min(1e-4)

@torch.inference_mode()
def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(1);data=load_data();x,y,_=data['val'];clean=clean_waveforms(y,619)
    cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.eval()
    def metrics(a,frontend_floor=None):
        if frontend_floor is None:logits=cnn(a.flatten(0,1))
        else:
            f=floor_features(cnn.encoder.frontend,a.flatten(0,1),frontend_floor)
            logits=cnn.head(cnn.encoder.blocks(cnn.encoder.stem(f)))
        p=logits.argmax(-1).reshape(-1,4);ok=p==y
        confusion=torch.bincount((y*10+p).flatten(),minlength=100).reshape(10,10)
        return {'digit_accuracy':float(ok.float().mean()),'sequence_accuracy':float(ok.all(-1).float().mean()),'confusion':confusion.tolist()}
    result={'scope':'validation only; original CNN weights unchanged','original_clean':metrics(clean),'original_corrupted':metrics(x),'ablations':{}}
    result['ablations']['pcm16_roundtrip']={'clean':metrics((clean*32767).to(torch.int16).float()/32768)}
    for drive in [.9,1.3,1.8]:
        result['ablations'][f'soft_clip_{drive}']={'clean':metrics(torch.tanh(clean*drive)/max(drive,1)),
                    'corrupted':metrics(torch.tanh(x*drive)/max(drive,1))}
    generator=torch.Generator().manual_seed(619)
    noise=torch.randn(clean.shape,generator=generator);mask=clean!=0;rms=clean.square().mean(-1,keepdim=True).sqrt()
    for ratio in [.0001,.001,.01,.03,.1]:
        result['ablations'][f'noise_ratio_{ratio}']={'clean':metrics(clean+ratio*rms*noise*mask)}
    for gamma in [1e-6,1e-5,1e-4,1e-3,1e-2]:
        result['ablations'][f'mel_floor_{gamma}']={'clean':metrics(clean,gamma),'corrupted':metrics(x,gamma)}
    frontend=cnn.encoder.frontend
    stats={}
    for name,a in [('clean',clean),('corrupted',x)]:
        v=a.flatten(0,1);spec=torch.stft(v,frontend.n_fft,frontend.hop_length,window=frontend.window,return_complex=True).abs().square()
        mel=torch.einsum('mf,bft->bmt',frontend.mel,spec)
        stats[name]={'fraction_mel_bins_at_log_floor':float((mel<=1e-6).float().mean()),
                     'median_positive_mel_energy':float(mel[mel>0].median()),'waveform_rms':float(v.square().mean().sqrt())}
    result['feature_statistics']=stats
    (ROOT/'clean_audit.json').write_text(json.dumps(result,indent=2))
    print('baseline',result['original_clean']['digit_accuracy'],result['original_corrupted']['digit_accuracy'])
    for name,v in result['ablations'].items():print(name,{k:m['digit_accuracy'] for k,m in v.items()})
    print(stats)

if __name__=='__main__':main()
