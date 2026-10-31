"""Local workflow API, separate from the packaged frontend. No training or writes."""
import argparse,base64,io,json,threading,time,wave
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
import torch
from .model import CNNBaseline,ModelConfig
from .appt import LocalAcousticTokenizer,LocalReranker,possibility_tree
from .encoder_study import ChannelProjection,ConditionedNCM,encoder_taps
from .safe_gate import RelativeFloorLogMel,BenefitGate,gate_features
from .data import synthesize_sequence,SynthesisConfig,DTMF_FREQUENCIES
ROOT=Path(__file__).resolve().parents[1]
LOCK=threading.Lock()


def make_audio(sequence,condition,strength,position,seed):
    if len(sequence)!=4 or any(d not in '0123456789' for d in sequence):raise ValueError('Enter exactly four digits, 0–9.')
    if condition not in ['clean','noise','partial_erasure','missing_frequency','interference']:raise ValueError('Unknown audio condition.')
    if not 0<=strength<=1 or not 0<=position<4 or not 0<=seed<2**32:raise ValueError('Settings out of range.')
    rng=np.random.default_rng(seed);wav,meta=synthesize_sequence(sequence,rng,SynthesisConfig(tone_ms=(180,180),gap_ms=(30,30)),corruption=False)
    events=np.stack([np.pad(wav[e['start_sample']:e['end_sample']],(0,960)) for e in meta['events']]).astype(np.float32)
    a=events[position,:1440]
    if condition=='noise':a+=rng.normal(0,1,len(a)).astype(np.float32)*np.sqrt(np.mean(a*a))*strength*3
    if condition=='partial_erasure':a[240:240+int(1200*strength)]=0
    if condition=='interference':a+=strength*.65*np.sin(2*np.pi*1336*np.arange(len(a))/8000).astype(np.float32)
    if condition=='missing_frequency':
        low,_=DTMF_FREQUENCIES[sequence[position]];a[:]=.36*np.sin(2*np.pi*low*np.arange(len(a))/8000+rng.uniform(0,2*np.pi))
    return torch.from_numpy(events).clamp(-1,1)


@lru_cache(maxsize=3)
def models(seed):
    if seed not in [317,419,523]:raise ValueError('Unknown checkpoint seed.')
    cfg=ModelConfig();cnn=CNNBaseline(cfg);cnn.load_state_dict(torch.load(ROOT/'representation_results/cnn.pt',weights_only=True));cnn.encoder.frontend=RelativeFloorLogMel(cfg,.001);cnn.eval()
    folder=ROOT/f'gate_results/seed_{seed}';meta=json.loads((ROOT/'gate_results/metrics.json').read_text())['seeds'][str(seed)]
    tok=LocalAcousticTokenizer();tok.load_state_dict(torch.load(folder/'acoustic_tokenizer.pt',weights_only=True));tok.freeze()
    local=LocalReranker();local.load_state_dict(torch.load(folder/'local.pt',weights_only=True));local.eval()
    gate=BenefitGate();gate.load_state_dict(torch.load(folder/'gate.pt',weights_only=True));gate.eval()
    adapter=ChannelProjection(64);adapter.load_state_dict(torch.load(ROOT/f'encoder_results/seed_{seed}/adapters.pt',weights_only=True)['final'])
    variants={};metrics=json.loads((ROOT/'decoder_results/metrics.json').read_text())['seeds'][str(seed)]['models']
    for name in ['original_joint','original_aligned','recognition']:
        m=ConditionedNCM(LocalAcousticTokenizer());path=ROOT/f'alignment_results/seed_{seed}/{name}.pt' if name!='recognition' else ROOT/f'decoder_results/seed_{seed}/recognition.pt'
        m.load_state_dict(torch.load(path,weights_only=True));m.eval();variants[name]=(m,metrics[name]['temperature'])
    return cnn,tok,local,gate,adapter,meta,variants


@torch.inference_mode()
def predict(events,seed,variant):
    """Audio only: target sequence and labels are deliberately absent."""
    cnn,tok,local,gate,adapter,meta,variants=models(seed)
    if variant not in variants:raise ValueError('Unknown model.')
    model,temp=variants[variant];start=time.perf_counter();taps=encoder_taps(cnn,events);z=adapter(taps['final'])
    base=(cnn.head(taps['final'])/meta['temperatures']['cnn']).softmax(-1)
    proxy=(local(tok.project(taps['final']))/meta['temperatures']['local']).softmax(-1)
    score=gate.score(gate_features(base,proxy,events.square().mean(-1).sqrt(),tok.project(taps['final'])))
    route=score>=meta['route_threshold'];accept=route&(score>=meta['accept_threshold']);cnn_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter();out=model(z);ncm_ms=(time.perf_counter()-start)*1000
    ncm=(out['digit_logits'][-1]/temp).softmax(-1);fused=base.clone();fused[accept]=ncm[accept];conventional=base.clone();conventional[accept]=proxy[accept]
    def ranking(p,mask):
        tree=possibility_tree(p,mask,5)
        return [{'sequence':''.join(map(str,row.tolist())),'probability':float(prob)} for row,prob in zip(tree['sequences'],tree['probabilities'])]
    # CNN exhaustive has all positions expanded. Protected methods share the same gate/list.
    ranks={'cnn':ranking(base,torch.ones(4,dtype=torch.bool)),'conventional':ranking(conventional,route),'ncm':ranking(fused,route)}
    return {'base':base.tolist(),'ncm':ncm.tolist(),'final':fused.tolist(),'route':route.tolist(),'accepted':accept.tolist(),'gate_score':score.tolist(),
      'codes':[c.tolist() for c in out['codes']], 'prefix_probabilities':[(v/temp).softmax(-1).tolist() for v in out['digit_logits']],
      'ranking':ranks,'candidate_count':10**int(route.sum()),'timing':{'cnn_gate_ms':cnn_ms,'ncm_ms':ncm_ms},
      'shapes':{k:list(v.shape) for k,v in taps.items()},'adapter_shape':list(z.shape),'seed':seed,'variant':variant}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT/'ui/lab'),**kwargs)
    def respond(self,obj,status=200):
        body=json.dumps(obj,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
    def do_GET(self):
        if self.path=='/api/status':return self.respond({'ready':True,'seeds':[317,419,523],'sample_rate':8000})
        if self.path=='/api/results':
            metrics=json.loads((ROOT/'decoder_results/metrics.json').read_text());out={}
            for name in ['original_joint','original_aligned','recognition']:
                rows=[v['models'][name]['cases']['corrupted']['generated']['protected'] for v in metrics['seeds'].values()]
                out[name]={k:float(np.mean([r[k] for r in rows])) for k in ['digit_accuracy','sequence_top1','sequence_top5']}
            return self.respond(out)
        return super().do_GET()
    def do_POST(self):
        if self.path!='/api/run':return self.respond({'error':'Unknown endpoint'},404)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=16384:raise ValueError('Invalid request size.')
            request=json.loads(self.rfile.read(size));seed=int(request.get('seed',317));sample_seed=int(request.get('audio_seed',42))
            events=make_audio(str(request.get('sequence','')),request.get('condition','clean'),float(request.get('strength',.5)),int(request.get('position',3)),sample_seed)
            with LOCK:out=predict(events,seed,request.get('variant','original_joint'))
            pcm=(events.flatten().clamp(-1,1).numpy()*32767).astype('<i2');buf=io.BytesIO()
            with wave.open(buf,'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(8000);f.writeframes(pcm.tobytes())
            out['audio']='data:audio/wav;base64,'+base64.b64encode(buf.getvalue()).decode();out['waveform']=events.flatten()[::16].tolist()
            return self.respond(out)
        except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:return self.respond({'error':str(exc)},400)
        except Exception as exc:
            import traceback;traceback.print_exc();return self.respond({'error':'Model service failed. Check the local server log.'},500)


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);a=p.parse_args();torch.set_num_threads(1)
    print(f'Possibility Lab: http://127.0.0.1:{a.port}',flush=True);ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
