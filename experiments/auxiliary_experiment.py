"""Capacity-aware auxiliary digit supervision with a matched continuation-free control."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from ppf.sequence import SequencePPF,mixture_scores
from ppf.compositional import CompositionalTokenizer
from ppf.model import CNNBaseline
from compositional_probes import CFG,load_data

ROOT=Path('auxiliary_results');C=json.loads(Path('experiments/auxiliary_config.json').read_text())

def tokenizer():
    tok=CompositionalTokenizer(CFG)
    tok.load_state_dict(torch.load('compositional_results/structured_tokenizer.pt',weights_only=True))
    return tok.freeze()

@torch.no_grad()
def target_prefixes(tok,codes):
    prefixes=[];levels=[]
    for book,code in zip(tok.books,codes):
        levels.append(book.lookup(code))
        prefixes.append(tok.decoder(sum(tok.resize(q,4) for q in levels)).softmax(-1))
    return prefixes


def auxiliary_losses(output,oracle_prefixes,labels):
    losses=[]
    for l,logits in enumerate(output['digit_logits']):
        if l<2:
            loss=F.kl_div(logits.log_softmax(-1),oracle_prefixes[l],reduction='none').sum(-1).mean()
        else:loss=F.cross_entropy(logits.flatten(0,1),labels.flatten())
        losses.append(loss)
    return losses

@torch.inference_mode()
def audit_order(model,x):
    model.eval();raw=model.encoder(x.flatten(0,1)).reshape(len(x),4,64,-1)+model.event_positions
    memory=model.audio_memory(x)
    restored=memory.reshape(len(x),64,4,-1).permute(0,2,1,3)
    direct=model.differentiable_generation(x);probs,codes,logits=model.generate(x,stochastic=False)
    return {'event_windows':list(x.shape),'cnn_event_features':list(raw.shape),'transformer_audio_memory':list(memory.shape),
            'reshape_preserves_event_order':bool(torch.equal(raw,restored)),
            'global_pooling_before_transformer':False,
            'straight_through_vs_inference_max_prob_difference':float((direct['digit_logits'][-1].softmax(-1)-probs[:,0]).abs().max()),
            'straight_through_vs_inference_codes_match':all(torch.equal(a,b) for a,b in zip(direct['codes'],codes))}

@torch.inference_mode()
def assess(model,data,cnn):
    model.eval();x,y,conditions=data;n=len(x)
    prefix_predictions=[];greedy_codes=[[],[],[]];oracle_predictions=[];scores=[];times=[];tfhit=np.zeros(3);totals=np.zeros(3)
    zero_decoded=[];swap_decoded=[];no_condition_decoded=[]
    for xx,yy in zip(x.split(32),y.split(32)):
        out=model.differentiable_generation(xx)
        prefix_predictions.append(torch.stack([l.argmax(-1) for l in out['digit_logits']],1))
        target=model.tokenizer(yy)['codes'];oracle=target_prefixes(model.tokenizer,target)
        oracle_predictions.append(torch.stack([p.argmax(-1) for p in oracle],1))
        tf=model.teacher_logits(xx,target)
        for l in range(3):
            greedy_codes[l].append(out['codes'][l]);tfhit[l]+=int((tf[l].argmax(-1)==target[l]).sum());totals[l]+=target[l].numel()
        fine=model.tokenizer.resize(model.tokenizer.books[2].lookup(out['codes'][2]),4)
        zero_decoded.append(model.tokenizer.decoder(fine).argmax(-1))
        shuffled=[c.roll(1,0) if l<2 else c for l,c in enumerate(out['codes'])]
        swap_decoded.append(model.tokenizer.decode_codes(shuffled).argmax(-1))
        # Sensitivity ablation: remove earlier embeddings from conditioning,
        # while keeping the resulting actual generated codes for decoding.
        memory=model.audio_memory(xx);levels=[];codes=[]
        for l,(length,book) in enumerate(zip(model.tokenizer.lengths,model.tokenizer.books)):
            logit=model.predictor.forward_level(memory,[torch.zeros_like(q) for q in levels],l,length)
            code=logit.argmax(-1);codes.append(code);levels.append(book.lookup(code))
        no_condition_decoded.append(model.tokenizer.decode_codes(codes).argmax(-1))
    pred=torch.cat(prefix_predictions);oracle=torch.cat(oracle_predictions)
    target=model.tokenizer(y)['codes'];gc=[torch.cat(c) for c in greedy_codes]
    cnn_pred=cnn(x.flatten(0,1)).argmax(-1).reshape(-1,4)
    # Timed production path only: no targets, diagnostic losses or ST operations.
    for _ in range(3):
        p,_,_=model.generate(x[:1],stochastic=False);mixture_scores(p).topk(5,-1)
    for sample in x:
        start=time.perf_counter();p,_,_=model.generate(sample[None],stochastic=False)
        score=mixture_scores(p);score.topk(5,-1);times.append((time.perf_counter()-start)*1000);scores.append(score)
    scores=torch.cat(scores);top=scores.topk(5,-1).indices;ids=y@torch.tensor([1000,100,10,1])
    ok=pred==y[:,None];oracle_ok=oracle==y[:,None]
    def metrics(mask):
        return {'samples':int(mask.sum()),'digit_accuracy_by_level_and_position':ok[mask].float().mean(0).tolist(),
                'sequence_accuracy_by_level':ok[mask].all(-1).float().mean(0).tolist(),
                'sequence_top1':float((top[mask,0]==ids[mask]).float().mean()),
                'sequence_top5':float((top[mask]==ids[mask,None]).any(-1).float().mean()),
                'sequence_nll':float(-scores[mask][torch.arange(int(mask.sum())),ids[mask]].clamp_min(1e-30).log().mean())}
    transitions=[]
    for l in range(2):
        before=ok[:,l];after=ok[:,l+1]
        transitions.append({'from_level':l+1,'to_level':l+2,'digit_wrong_to_correct_by_position':((~before)&after).sum(0).tolist(),
              'digit_correct_to_wrong_by_position':(before&(~after)).sum(0).tolist(),
              'sequence_wrong_to_correct':int(((~before.all(-1))&after.all(-1)).sum()),
              'sequence_correct_to_wrong':int((before.all(-1)&(~after.all(-1))).sum())})
    zero=torch.cat(zero_decoded);swapped=torch.cat(swap_decoded);no_condition=torch.cat(no_condition_decoded)
    def contribution(p):
        return {'digit_accuracy':float((p==y).float().mean()),'sequence_accuracy':float((p==y).all(-1).float().mean()),
                'digit_decisions_changed':int((p!=pred[:,-1]).sum()),
                'digits_wrong_to_correct':int(((pred[:,-1]!=y)&(p==y)).sum()),
                'digits_correct_to_wrong':int(((pred[:,-1]==y)&(p!=y)).sum())}
    return {'overall':metrics(torch.ones(n,dtype=torch.bool)),
            'by_condition':{c:metrics(torch.tensor([v==c for v in conditions])) for c in sorted(set(conditions))},
            'cnn_digit_accuracy_by_position':(cnn_pred==y).float().mean(0).tolist(),
            'generated_code_accuracy_by_level':[float((a==b).float().mean()) for a,b in zip(gc,target)],
            'generated_fine_code_accuracy_by_position':(gc[2]==target[2]).float().mean(0).tolist(),
            'teacher_code_accuracy_by_level':(tfhit/totals).tolist(),
            'oracle_prefix_digit_accuracy_by_level_and_position':oracle_ok.float().mean(0).tolist(),
            'transitions':transitions,
            'coarse_contribution':{'fine_only_decoder':contribution(zero),'swap_coarse_and_middle_decoder':contribution(swapped),
                                   'zero_previous_level_conditioning':contribution(no_condition)},
            'latency_ms':{'p50':float(np.percentile(times,50)),'p95':float(np.percentile(times,95))},
            'audit':audit_order(model,x[:8])}


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(C['threads']);torch.use_deterministic_algorithms(True)
    (ROOT/'config.json').write_text(json.dumps(C,indent=2));data=load_data()
    cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.eval()
    results={'config':C,'runs':{}}
    # Audit the exact previous structured checkpoint before any new training.
    existing=SequencePPF(tokenizer(),CFG);existing.load_state_dict(torch.load('compositional_results/structured_generator.pt',weights_only=True))
    results['existing_structured_reference']=assess(existing,data['test'],cnn)
    (ROOT/'comparison.json').write_text(json.dumps(results,indent=2))
    for name,use_aux in [('matched_baseline',False),('auxiliary',True)]:
        tok=tokenizer();torch.manual_seed(C['seed']);model=SequencePPF(tok,CFG)
        model.encoder.load_state_dict(cnn.encoder.state_dict())
        for p in model.encoder.parameters():p.requires_grad_(False)
        before={k:v.clone() for k,v in tok.state_dict().items()}
        encoder_before={k:v.clone() for k,v in model.encoder.state_dict().items()}
        loader=DataLoader(TensorDataset(data['train'][0],data['train'][1]),batch_size=C['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(C['seed']))
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=C['learning_rate'])
        history=[];gradient_checks=[];begin=time.perf_counter()
        for epoch in range(C['epochs']):
            model.train();total=np.zeros(5)
            for step,(x,y) in enumerate(loader):
                with torch.no_grad():
                    codes=tok(y)['codes'];oracle=target_prefixes(tok,codes)
                opt.zero_grad(set_to_none=True)
                teacher=model.teacher_logits(x,codes)
                code_loss=sum(F.cross_entropy(l.flatten(0,1),c.flatten()) for l,c in zip(teacher,codes))/3
                # Both arms consume identical dropout draws in this extra pass.
                output=model.differentiable_generation(x)
                aux=auxiliary_losses(output,oracle,y)
                weighted=sum(w*l for w,l in zip(C['auxiliary_weights'],aux))/sum(C['auxiliary_weights'])
                loss=code_loss+weighted if use_aux else code_loss
                if use_aux and step==0 and epoch==0:
                    # Isolate supervision-to-generator gradients before total backward.
                    for l,aloss in enumerate(aux):
                        g=torch.autograd.grad(aloss,output['code_logits'][l],retain_graph=True)[0]
                        gradient_checks.append({'level':l+1,'auxiliary_loss':float(aloss.detach()),'gradient_norm_to_same_level_code_logits':float(g.norm())})
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                total+=np.array([float(code_loss.detach()),*[float(a.detach()) for a in aux],float(norm)])
            row={'epoch':epoch+1,'code_ce':total[0]/len(loader),'aux_losses':(total[1:4]/len(loader)).tolist(),'gradient_norm_before_clip':total[4]/len(loader)}
            history.append(row);print(name,json.dumps(row),flush=True)
        assert all(torch.equal(v,before[k]) for k,v in tok.state_dict().items())
        assert all(torch.equal(v,encoder_before[k]) for k,v in model.encoder.state_dict().items())
        row={'history':history,'training_seconds':time.perf_counter()-begin,'auxiliary_gradient_checks':gradient_checks,
             'tokenizer_unchanged':True,'cnn_unchanged':True,'optimizer_steps':len(loader)*C['epochs'],
             'total_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
        torch.save(model.state_dict(),ROOT/f'{name}.pt')
        for split in ['val','test']:row[split]=assess(model,data[split],cnn)
        results['runs'][name]=row;(ROOT/'comparison.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({'arm':name,'test':row['test']['overall'],'codes':row['test']['generated_code_accuracy_by_level']}),flush=True)
    paths=['ppf/sequence.py','experiments/auxiliary_config.json','compositional_results/structured_tokenizer.pt','representation_results/cnn.pt']
    (ROOT/'source_hashes.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},indent=2))

if __name__=='__main__':main()
