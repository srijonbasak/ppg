import json,hashlib
from pathlib import Path
old=json.loads(Path('alignment_results/metrics.json').read_text());new=json.loads(Path('decoder_results/metrics.json').read_text());verified={}
for seed in ['317','419','523']:
 assert new['seeds'][seed]['training_trigger']
 assert set(new['seeds'][seed]['models'])=={'original_joint','original_aligned','recognition'}
 for model in ['original_joint','original_aligned']:
  for case in ['corrupted','clean','noisy','partial_erasure','missing_frequency']:
   a=old['seeds'][seed]['models'][model]['scenarios'][case];b=new['seeds'][seed]['models'][model]['cases'][case]['generated']
   for kind in ['raw','protected']:
    for key in ['digit_accuracy','sequence_top1','sequence_top5','digit_nll']:
     assert abs(a[kind][key]-b[kind][key])<1e-6,(seed,model,case,kind,key)
 for case in ['corrupted','clean','noisy','partial_erasure','missing_frequency']:
  a=Path(f'encoder_results/seed_{seed}/{case}_branches.json');b=Path(f'decoder_results/seed_{seed}/{case}_branches.json');assert a.read_bytes()==b.read_bytes()
for f,h in json.loads(Path('alignment_results/source_manifest.json').read_text()).items():
 if f!='checkpoint_tensor_identity_verified_seeds':assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
for p in [Path('ppf/decoder_audit.py'),Path('experiments/decoder_sensitivity.py'),Path('experiments/decoder_detail.py')]:verified[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
verified['checks']=['all 3 seeds and 5 conditions complete','baseline replay equals preserved metrics (tolerance 1e-6)','candidate policy files identical','preserved source/checkpoint manifest unchanged']
Path('decoder_results/verification.json').write_text(json.dumps(verified,indent=2));print('All result, baseline, candidate and preservation checks passed.')
