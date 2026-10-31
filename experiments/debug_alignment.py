"""No-training alignment, ordering and feature-information controls."""
import hashlib
import json
from pathlib import Path
import torch
from ppf.model import CNNBaseline
from debug_conditional import ROOT,CFG,load_model

def main():
    torch.set_num_threads(1)
    data=torch.load('representation_results/audio_data.pt',weights_only=False)
    rows=[json.loads(s) for s in Path('representation_results/audio_manifest.jsonl').read_text().splitlines()]
    hashes=set()
    for row in rows:
        audio,labels,conditions=data[row['split']];i=row['index']
        digest=hashlib.sha256(audio[i].numpy().tobytes()).hexdigest()
        assert digest==row['sha256'] and labels[i].tolist()==row['digits'] and conditions[i]==row['condition']
        assert digest not in hashes;hashes.add(digest)
    x,y,_=data['test'];result={'manifest_tensor_label_alignment':True,'checked_sequences':len(rows),'models':{}}
    with torch.inference_mode():
        for name,path in [('original','representation_results/ppf.pt'),
                          ('teacher_continuation',ROOT/'teacher_continuation.pt'),
                          ('mixed_prefix',ROOT/'mixed_prefix.pt')]:
            model=load_model();model.load_state_dict(torch.load(path,weights_only=True));model.eval()
            a=model.predictor.forward_level(model.audio_memory(x),[],0,1)
            b=model.predictor.forward_level(model.audio_memory(x.roll(1,1)),[],0,1)
            target=model.tokenizer(y)['codes'][0]
            changed=model.tokenizer(y.roll(1,1))['codes'][0]
            result['models'][name]={
              'coarse_prediction_unchanged_after_event_rotation':float((a.argmax(-1)==b.argmax(-1)).float().mean()),
              'coarse_target_unchanged_after_event_rotation':float((target==changed).float().mean()),
              'coarse_accuracy_rotated':float((b.argmax(-1)==changed).float().mean()),
              'coarse_logit_mean_abs_rotation_change':float((a-b).abs().mean()),
              'coarse_logit_std':float(a.std()),
              'event_position_rms':float(model.event_positions.square().mean().sqrt())}
        # Diagnostic bridge: use CNN PREDICTED digits, not correct labels, to
        # derive codes. This bypasses the Transformer and is not a PPF result.
        cnn=CNNBaseline(CFG);cnn.load_state_dict(torch.load('representation_results/cnn.pt',weights_only=True));cnn.eval()
        predicted=cnn(x.flatten(0,1)).argmax(-1).reshape(-1,4)
        target=model.tokenizer(y)['codes'];bridge=model.tokenizer(predicted)['codes']
        result['cnn_predicted_label_bridge']={
          'scope':'inference from CNN predicted digits; diagnostic bypass of generator, no ground-truth prediction inputs',
          'digit_accuracy':float((predicted==y).float().mean()),
          'sequence_accuracy':float((predicted==y).all(-1).float().mean()),
          'code_accuracy_by_level':[float((a==b).float().mean()) for a,b in zip(target,bridge)]}
    (ROOT/'alignment_controls.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
