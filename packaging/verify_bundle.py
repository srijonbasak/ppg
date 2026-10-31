import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
files=['representation_results/cnn.pt','gate_results/metrics.json','decoder_results/metrics.json','ui/lab/index.html','ui/lab/app.js','ui/lab/style.css']
for seed in [317,419,523]:
 files += [f'gate_results/seed_{seed}/acoustic_tokenizer.pt',f'gate_results/seed_{seed}/local.pt',f'gate_results/seed_{seed}/gate.pt',f'encoder_results/seed_{seed}/adapters.pt',f'alignment_results/seed_{seed}/original_joint.pt',f'alignment_results/seed_{seed}/original_aligned.pt',f'decoder_results/seed_{seed}/recognition.pt']
result={}
for platform in ['linux','windows']:
 bundle=root/f'packaging/dist-{platform}/PossibilityLab/_internal'
 result[platform]={}
 for f in files:
  a=hashlib.sha256((root/f).read_bytes()).hexdigest();b=hashlib.sha256((bundle/f).read_bytes()).hexdigest()
  assert a==b,(platform,f)
  result[platform][f]=a
(root/'releases/bundled-resources.json').write_text(json.dumps(result,indent=2));print('Both bundles contain all checkpoint and UI files with matching source hashes.')
