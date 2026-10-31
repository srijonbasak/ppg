"""Summarize saved comparison and measure tokenizer/generator diagnostics."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ppf.engine import ToneEventDataset, load_checkpoint

torch.set_num_threads(1)
root=Path('audit_results')
d=json.loads((root/'comparison.json').read_text())
ds=ToneEventDataset(root/'data/manifest.jsonl','test')
diagnostics=[]
with torch.inference_mode():
    for seed in d['training_seeds']:
        model,_=load_checkpoint(root/f'ppf_{seed}.pt'); model.eval()
        n=0; teacher=np.zeros(3); generated=np.zeros(3); mse=np.zeros(3)
        recover=np.zeros(2); regress=np.zeros(2); codes=[set(),set(),set()]
        for audio,labels in DataLoader(ds,batch_size=32):
            latent=model.encoder(audio); tok=model.tokenizer(latent)
            target=model.tokenizer.proj_in(latent).transpose(1,2)
            accum=torch.zeros_like(target); gen=torch.zeros_like(target)
            _,indices=model.predict_latent(audio)
            previous=None
            for level in range(3):
                accum+=model.tokenizer._resize(tok.quantized_levels[level],target.shape[1])
                gen+=model.tokenizer._resize(model.tokenizer.codebooks[level].embedding(indices[level]),target.shape[1])
                ok=model.decoder(gen).argmax(-1)==labels
                teacher[level]+=int((model.decoder(accum).argmax(-1)==labels).sum())
                generated[level]+=int(ok.sum())
                mse[level]+=float((accum-target).square().mean())*len(labels)
                codes[level].update(tok.codes[level].flatten().tolist())
                if previous is not None:
                    recover[level-1]+=int((~previous & ok).sum())
                    regress[level-1]+=int((previous & ~ok).sum())
                previous=ok
            n+=len(labels)
        diagnostics.append({'seed':seed,'events':n,
          'teacher_audio_tokenizer_digit_accuracy_by_level':(teacher/n).tolist(),
          'teacher_audio_latent_mse_by_level':(mse/n).tolist(),
          'greedy_generated_digit_accuracy_by_level':(generated/n).tolist(),
          'wrong_to_correct_counts':recover.tolist(),'correct_to_wrong_counts':regress.tolist(),
          'test_codebook_unique_codes':[len(v) for v in codes]})
(root/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
lines=['# Measured comparison', '', 'Mean ± sample standard deviation across three training seeds; same 120 test sequences.', '',
'| Model | Digit accuracy | Sequence top-1 | Sequence top-5 | Sequence NLL | CPU p50 / p95 ms |',
'|---|---:|---:|---:|---:|---:|']
for name,m in [('cnn',1),('ppf',1),('ppf',4),('ppf',8),('ppf',16)]:
    runs=[r['metrics'] for r in d['runs'] if r['model']==name and r['metrics']['hierarchy_samples']==m]
    def fmt(key,percent=False):
        a=np.array([r[key] for r in runs])*(100 if percent else 1)
        return f'{a.mean():.2f} ± {a.std(ddof=1):.2f}'+('%' if percent else '')
    latency=' / '.join(f'{np.mean([r["latency_ms"][p] for r in runs]):.2f}' for p in ['p50','p95'])
    lines.append(f'| {name.upper()}'+(f' M={m}' if name=='ppf' else '')+f' | {fmt("digit_accuracy",True)} | {fmt("sequence_top1",True)} | {fmt("sequence_top5",True)} | {fmt("sequence_nll")} | {latency} |')
lines+=['','Latency columns average per-run percentiles; they are not pooled percentiles. GPU unavailable.', '',
'## Reconstruction and level diagnostics', '', 'These concern the existing audio tokenizer, not the specified clean-label tokenizer. Greedy generated-prefix accuracy is diagnostic; the main comparison uses stochastic sampling.', '']
for row in diagnostics:
    lines.append(f'- Seed {row["seed"]}: teacher digit accuracy by level {row["teacher_audio_tokenizer_digit_accuracy_by_level"]}; latent MSE {row["teacher_audio_latent_mse_by_level"]}; greedy generated accuracy {row["greedy_generated_digit_accuracy_by_level"]}; unique codes {row["test_codebook_unique_codes"]}.')
lines+=['','The current PPF did not show a benefit in this fixed short training protocol. This is not evidence against the finalized architecture, which is not implemented. CNN reached a ceiling on this synthetic distribution. Training logs show severe PPF loss growth; causes have not been isolated. No hyperparameters were changed after seeing test results.', '',
'Full numerical outputs: comparison.json and diagnostics.json. Scope, audit findings, reproducibility and limitations: AUDIT.md.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
