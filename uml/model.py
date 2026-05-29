"""Dual-branch UML model: EMG + audio sharing a single Transformer.

Branches
--------
EMG   branch:  ``GaddyRawEMGEncoder``                       → emg_ctc_head
                (CNN frontend + transformer)
Audio branch:  ``AudioFrontend`` (frozen wav2vec2 + linear) → audio_ctc_head
                ↓
                SAME ``GaddyRawEMGEncoder.transformer`` instance

The transformer is literally shared: ``model.emg_encoder.transformer`` is the
SAME ``nn.Module`` evaluated on the audio path. AudioFrontend (wav2vec2-base)
is always frozen — only its projection head is learnable.

CTC heads
---------
Configurable: ``share_ctc_head=True`` makes ``emg_ctc_head`` and
``audio_ctc_head`` the SAME ``CTCHead`` instance; otherwise each branch has
its own readout (default — matches the reference implementation).

Inference uses the EMG branch only — ``model(raw_emg)`` delegates to
``forward_emg``. After UML training, the EMG-branch state dict
(``emg_encoder.*`` + ``emg_ctc_head.*``) maps directly onto a baseline
``BaselineCTCModel`` for fine-tuning.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from semg_jepa.architecture import CTCHead, GaddyRawEMGEncoder


# ---------------------------------------------------------------------------
# AudioFrontend — frozen wav2vec2-base + trainable projection
# ---------------------------------------------------------------------------

class AudioFrontend(nn.Module):
    """Encodes raw 16-kHz waveforms with a frozen wav2vec2-base; projects the
    768-dim hidden states to ``model_size`` so they can be fed into the shared
    Transformer.

    Input:  ``waveform`` (B, T_audio) — zero-mean / unit-variance normalized
    Output: ``(features (B, T', model_size), out_lengths (B,))``

    The wav2vec2 feature extractor downsamples by ~320×: a 16 kHz, 1-second
    waveform yields ~50 frames at the projection output.
    """

    WAV2VEC2_MODEL = "facebook/wav2vec2-base"

    def __init__(self, model_size: int = 768):
        super().__init__()
        from transformers import Wav2Vec2Model  # local import to avoid hard dep at module load

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(self.WAV2VEC2_MODEL)
        for p in self.wav2vec2.parameters():
            p.requires_grad = False
        wav2vec2_dim = self.wav2vec2.config.hidden_size
        self.projection = nn.Linear(wav2vec2_dim, model_size)

    def _attention_mask(self, waveform: torch.Tensor, audio_lengths: torch.Tensor | None):
        if audio_lengths is None:
            return None
        B, T = waveform.shape
        return (
            torch.arange(T, device=waveform.device).unsqueeze(0)
            < audio_lengths.unsqueeze(1)
        ).long()

    def _output_lengths(self, audio_lengths: torch.Tensor | None, T_prime: int, batch_size: int, device) -> torch.Tensor:
        if audio_lengths is None:
            return torch.full((batch_size,), T_prime, dtype=torch.long, device=device)
        # wav2vec2-base feature extractor: ~320× downsample with first-conv 400-tap window.
        if hasattr(self.wav2vec2, "_get_feat_extract_output_lengths"):
            out = self.wav2vec2._get_feat_extract_output_lengths(audio_lengths)
        else:
            out = (audio_lengths - 400) // 320 + 1
        return out.long().clamp(min=1, max=T_prime)

    def forward(self, waveform: torch.Tensor, audio_lengths: torch.Tensor | None = None):
        attention_mask = self._attention_mask(waveform, audio_lengths)
        with torch.no_grad():
            outputs = self.wav2vec2(input_values=waveform, attention_mask=attention_mask)
        features = outputs.last_hidden_state                                      # (B, T', 768)
        out_lengths = self._output_lengths(
            audio_lengths, features.shape[1], waveform.shape[0], waveform.device,
        )
        return self.projection(features), out_lengths                              # (B, T', model_size)


# ---------------------------------------------------------------------------
# UMLModel
# ---------------------------------------------------------------------------

class UMLModel(nn.Module):
    """Dual-branch UML model. The shared Transformer is the SAME Python object
    on both paths (``self.emg_encoder.transformer``).

    The EMG branch is a vanilla :class:`GaddyRawEMGEncoder` so its weights map
    1-to-1 onto the baseline pipeline at fine-tune time. The audio branch
    plugs ``AudioFrontend`` outputs into the same transformer.
    """

    def __init__(
        self,
        vocab_size: int = 37,
        model_size: int = 768,
        num_layers: int = 6,
        dropout: float = 0.2,
        share_ctc_head: bool = False,
    ):
        super().__init__()
        self.emg_encoder = GaddyRawEMGEncoder(
            model_size=model_size, num_layers=num_layers, dropout=dropout,
        )
        self.audio_frontend = AudioFrontend(model_size=model_size)

        self.share_ctc_head = bool(share_ctc_head)
        if self.share_ctc_head:
            head = CTCHead(model_size=model_size, vocab_size=vocab_size)
            self.emg_ctc_head = head
            self.audio_ctc_head = head
        else:
            self.emg_ctc_head = CTCHead(model_size=model_size, vocab_size=vocab_size)
            self.audio_ctc_head = CTCHead(model_size=model_size, vocab_size=vocab_size)

        self.blank_id = vocab_size  # CTCHead outputs vocab_size+1 logits, blank = last index

    # ------------------------------------------------------------------
    # EMG branch
    # ------------------------------------------------------------------

    def forward_emg(self, raw_emg: torch.Tensor) -> torch.Tensor:
        latent = self.emg_encoder(raw_emg)        # (B, T_raw // 8, D)
        return self.emg_ctc_head(latent)          # (B, T, vocab+1) — raw logits

    # ------------------------------------------------------------------
    # Audio branch — uses the SAME ``self.emg_encoder.transformer`` instance
    # ------------------------------------------------------------------

    def forward_audio(
        self,
        waveform: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, out_lengths = self.audio_frontend(waveform, audio_lengths)   # (B, T', D)
        # Transformer expects (T, B, D) — same convention as GaddyRawEMGEncoder.
        x = self.emg_encoder.transformer(x.transpose(0, 1)).transpose(0, 1)
        logits = self.audio_ctc_head(x)
        return logits, out_lengths

    # ------------------------------------------------------------------
    # Convenience: inference uses EMG branch only
    # ------------------------------------------------------------------

    def forward(self, raw_emg: torch.Tensor) -> torch.Tensor:
        return self.forward_emg(raw_emg)


# ---------------------------------------------------------------------------
# CTC loss helper (works on either branch's logits)
# ---------------------------------------------------------------------------

def ctc_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
) -> torch.Tensor:
    """``logits``: (B, T, V+1). Applies log-softmax then F.ctc_loss with the
    sequence-first layout F.ctc_loss expects.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1).contiguous()  # (T, B, V+1)
    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="mean",
        zero_infinity=True,
    )
