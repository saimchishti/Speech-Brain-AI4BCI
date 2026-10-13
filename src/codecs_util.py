"""Mimi (kyutai/mimi) and HiFi-GAN (speechbrain/tts-hifigan-libritts-16kHz) wrappers."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('HF_HOME', os.path.join(ROOT, 'checkpoints/cache/hf'))
import torch, torch.nn.functional as F, torchaudio

NQ = 8          # Mimi codebooks used (Moshi setting): level 1 semantic (WavLM-distilled), 2-8 acoustic
DEV = 'cuda'


class Mimi:
    def __init__(self, dev=DEV):
        from transformers import MimiModel
        self.m = MimiModel.from_pretrained('kyutai/mimi').to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def latents(self, wav24):
        """wav24 (B, N) float -> pre-quantization latent (B, 512, T) at 12.5 Hz"""
        x = wav24.to(self.dev)[:, None]
        e = self.m.encoder(x)
        e = self.m.encoder_transformer(e.transpose(1, 2), return_dict=True).last_hidden_state.transpose(1, 2)
        return self.m.downsample(e)

    @torch.no_grad()
    def codes(self, wav24):
        """-> (B, NQ, T) int"""
        return self.m.quantizer.encode(self.latents(wav24), NQ).transpose(0, 1)

    @torch.no_grad()
    def _dec(self, emb):
        e = self.m.upsample(emb)
        e = self.m.decoder_transformer(e.transpose(1, 2), return_dict=True).last_hidden_state.transpose(1, 2)
        return self.m.decoder(e)[:, 0]

    @torch.no_grad()
    def decode_codes(self, codes):
        return self._dec(self.m.quantizer.decode(codes.to(self.dev)))

    @torch.no_grad()
    def decode_latents(self, lat):
        """Unquantized path (Mimi is trained with quantization bypassed 50% of the time): x -> in_proj -> out_proj, both RVQs."""
        q = self.m.quantizer
        s, a = q.semantic_residual_vector_quantizer, q.acoustic_residual_vector_quantizer
        lat = lat.to(self.dev)
        return self._dec(s.output_proj(s.input_proj(lat)) + a.output_proj(a.input_proj(lat)))


class HiFiGAN:
    """80-bin mel per the model card: sr 16k, n_fft 1024, win 1024, hop 256, fmin 0, fmax 8000, power 1, slaney, log-compressed."""
    def __init__(self, dev=DEV):
        from speechbrain.inference.vocoders import HIFIGAN
        self.v = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-libritts-16kHz',
                                      savedir=os.path.join(ROOT, 'checkpoints/cache/sb_hifigan'), run_opts={'device': dev})
        self.mel_t = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, win_length=1024, hop_length=256,
                                                          f_min=0., f_max=8000., n_mels=80, power=1, norm='slaney',
                                                          mel_scale='slaney', center=True).to(dev)
        self.dev = dev

    @torch.no_grad()
    def mel(self, wav16):
        """(B, N) -> (B, 80, T) log mel (dynamic range compression as in speechbrain: log(clamp(x, 1e-5)))"""
        return torch.log(torch.clamp(self.mel_t(wav16.to(self.dev)), min=1e-5))

    @torch.no_grad()
    def wav(self, mel):
        # generator pads `inference_padding`=5 frames on each side -> drop 5*256 leading samples to re-align
        w = self.v.decode_batch(mel.to(self.dev))[:, 0]
        return w[:, 5 * 256: 5 * 256 + mel.shape[-1] * 256]
