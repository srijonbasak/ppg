"""Validation-trained routing and acceptance for local APPT refinement."""
import torch
from torch import nn
import torch.nn.functional as F
from .features import LogMelSpectrogram
from .appt import possibility_tree,uncertainty_gate


class RelativeFloorLogMel(LogMelSpectrogram):
    def __init__(self,cfg,relative_floor=.001):
        super().__init__(cfg.sample_rate,cfg.n_fft,cfg.hop_length,cfg.n_mels)
        self.relative_floor=relative_floor

    def forward(self,waveform):
        if waveform.ndim==1:waveform=waveform[None]
        spec=torch.stft(waveform,self.n_fft,self.hop_length,window=self.window,return_complex=True).abs().square()
        mel=torch.einsum('mf,bft->bmt',self.mel,spec)
        log=(mel+self.relative_floor*mel.mean((-2,-1),keepdim=True)).clamp_min(1e-6).log()
        return (log-log.mean((-2,-1),keepdim=True))/log.std((-2,-1),keepdim=True).clamp_min(1e-4)


def gate_features(base,candidate,rms,latent):
    def summary(p):
        v=p.topk(2,-1).values
        return [v[:,0],-(p*p.clamp_min(1e-12).log()).sum(-1)/2.302585093,v[:,0]-v[:,1]]
    mid=(base+candidate)/2
    js=.5*((base*(base.clamp_min(1e-12)/mid.clamp_min(1e-12)).log()).sum(-1)+(candidate*(candidate.clamp_min(1e-12)/mid.clamp_min(1e-12)).log()).sum(-1))
    i=torch.arange(len(base),device=base.device)
    columns=summary(base)+summary(candidate)+[(base*candidate).sum(-1),js,
        (base.argmax(-1)!=candidate.argmax(-1)).float(),rms.clamp_min(1e-10).log(),
        latent.square().mean((-2,-1)).sqrt(),base[i,candidate.argmax(-1)],candidate[i,base.argmax(-1)]]
    return torch.stack(columns,-1)


class BenefitGate(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('mean',torch.zeros(13));self.register_buffer('std',torch.ones(13))
        self.linear=nn.Linear(13,3)
    def forward(self,features):return self.linear((features-self.mean)/self.std)
    def score(self,features):
        p=self(features).softmax(-1)
        return p[:,2]-p[:,0]


class SafeAdaptiveTree:
    """Separate a cheap route decision from acceptance of the actual proposal.

    The same gate, features and thresholds serve all proposal mechanisms.
    Only audio and optional fault-injected probabilities enter prediction.
    """
    def __init__(self,cnn,tokenizer,verifier,refiner,temperatures,gate,route_cutoff,accept_cutoff,old_cutoff):
        self.cnn=cnn.eval();self.tokenizer=tokenizer.freeze();self.verifier=verifier.eval();self.refiner=refiner.eval()
        self.temperatures=temperatures;self.gate=gate.eval();self.route_cutoff=route_cutoff;self.accept_cutoff=accept_cutoff;self.old_cutoff=old_cutoff

    @torch.inference_mode()
    def predict(self,events,method='appt_gated',base_override=None,top_k=5):
        if method not in ('cnn','cnn_gated','local_old','appt_old','local_gated','appt_gated'):
            raise ValueError('Unknown method')
        features=self.cnn.encoder(events);p=(self.cnn.head(features)/self.temperatures['cnn']).softmax(-1)
        if base_override is not None:p=base_override.clone()
        erased=events.square().mean(-1)<=1e-12;p[erased]=.1
        route=torch.zeros(4,dtype=torch.bool);accept=route.clone();refined=0
        result=p.clone()
        if method!='cnn':
            z=self.tokenizer.project(features);rms=events.square().mean(-1).sqrt()
            proxy=(self.verifier(z)/self.temperatures['local']).softmax(-1);proxy[erased]=.1
            if method.endswith('_old'):route,_=uncertainty_gate(p,proxy,self.old_cutoff,.8)
            else:route=self.gate.score(gate_features(p,proxy,rms,z))>=self.route_cutoff
            route=route|erased;chosen=route&(~erased)
            candidate=p.clone()
            if method.startswith('local'):candidate[chosen]=proxy[chosen];refined=int(chosen.sum())
            elif method.startswith('appt') and chosen.any():
                candidate[chosen]=(self.refiner(z[chosen])['digit_logits'][-1]/self.temperatures['appt']).softmax(-1)
                refined=int(chosen.sum())
            if method=='cnn_gated':accept=torch.zeros_like(route)
            elif method.endswith('_old'):accept=chosen
            else:accept=chosen&(self.gate.score(gate_features(p,candidate,rms,z))>=self.accept_cutoff)
            result[accept]=candidate[accept]
        result[erased]=.1
        tree=possibility_tree(result,torch.ones_like(route) if method=='cnn' else route,top_k)
        return {'digit_probabilities':result,'base_probabilities':p,'expanded':route,'accepted':accept,
                'refined_positions':refined,'erased':erased,**tree}
