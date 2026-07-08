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
    features to ``model_size`` so they can be fed into the shared Transformer.

    Input:  ``waveform`` (B, T_audio) — zero-mean / unit-variance normalized
    Output: ``(features (B, T', model_size), out_lengths (B,))``

    The wav2vec2 feature extractor downsamples by ~320×: a 16 kHz, 1-second
    waveform yields ~50 frames at the projection output.

    ``mode`` controls **how much of wav2vec2 runs before the shared transformer**
    — a knob for frontend symmetry with the conv-only EMG branch:

    * ``full`` (default) — the whole wav2vec2 (7-layer conv feature extractor **+
      12 self-attention layers**); we take ``last_hidden_state`` (768-dim). Audio
      thus reaches the shared transformer already contextualized by attention,
      while EMG has had none — an asymmetry that may cap UML transfer.
    * ``conv`` — the conv **feature extractor only** (``conv_dim[-1]`` = 512),
      skipping wav2vec2's 12 attention layers. Audio then reaches the shared
      transformer with NO self-attention applied yet, mirroring the conv-only EMG
      frontend, so the shared transformer does the contextualization for both.
    """

    WAV2VEC2_MODEL = "facebook/wav2vec2-base"

    def __init__(self, model_size: int = 768, mode: str = "full"):
        super().__init__()
        if mode not in ("full", "conv"):
            raise ValueError(f"AudioFrontend mode must be 'full' or 'conv', got {mode!r}")
        self.mode = mode
        from transformers import Wav2Vec2Model  # local import to avoid hard dep at module load

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(self.WAV2VEC2_MODEL)
        for p in self.wav2vec2.parameters():
            p.requires_grad = False
        in_dim = (
            self.wav2vec2.config.hidden_size if mode == "full"
            else self.wav2vec2.config.conv_dim[-1]
        )
        self.projection = nn.Linear(in_dim, model_size)

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
        if self.mode == "conv":
            # conv feature extractor only (no attention); (B, 512, T') -> (B, T', 512).
            with torch.no_grad():
                features = self.wav2vec2.feature_extractor(waveform).transpose(1, 2)
        else:
            attention_mask = self._attention_mask(waveform, audio_lengths)
            with torch.no_grad():
                outputs = self.wav2vec2(input_values=waveform, attention_mask=attention_mask)
            features = outputs.last_hidden_state                                  # (B, T', 768)
        # T' is set by the conv feature extractor in BOTH modes (the 12 attention
        # layers preserve length), so out_lengths is computed the same way.
        out_lengths = self._output_lengths(
            audio_lengths, features.shape[1], waveform.shape[0], waveform.device,
        )
        return self.projection(features), out_lengths                              # (B, T', model_size)


# ---------------------------------------------------------------------------
# TextFrontend — trainable char embedding (phase-2 text branch)
# ---------------------------------------------------------------------------

class TextFrontend(nn.Module):
    """Maps a (corrupted) character-id stream to ``model_size`` so it can feed the
    shared Transformer. Mirrors :class:`AudioFrontend`'s contract:
    ``forward(token_ids, token_lengths) -> (features (B, T, D), out_lengths (B,))``.

    Two modes (``mode=``):

    * ``embed`` (default) — trainable ``nn.Embedding`` + light projection. The
      denoising corruption AND the jittered upsample both happen in the dataset
      collate (``uml.text_dataset``); this frontend is a pure embedding.

    * ``frozen`` — **architectural mirror of** :class:`AudioFrontend`: a FROZEN
      pretrained character encoder (CANINE, the text analogue of frozen
      wav2vec2) → trainable ``Linear`` projection. Only the projection learns.
      Here the corrupted ids are fed un-upsampled to the frozen encoder (so it
      sees natural text), and the *features* are upsampled ×``upsample`` after the
      projection (so the dataset must NOT pre-upsample in this mode). This puts
      the branch at: frozen encoder → linear → shared transformer → CTC head,
      exactly like the audio branch.

    Input ids live in ``[0, vocab_size]``: ``0..vocab_size-1`` are real chars and
    index ``vocab_size`` is a ``MASK`` token used **only on the input side**
    (distinct from the CTC blank ``vocab_size`` in output logit space).
    """

    CANINE_MODEL = "google/canine-s"
    BYT5_MODEL = "google/byt5-small"

    def __init__(self, vocab_size: int = 37, model_size: int = 768, dropout: float = 0.2,
                 mode: str = "embed", frozen_model: str | None = None, frozen_arch: str = "canine",
                 upsample: int = 3, chars: str | None = None):
        super().__init__()
        if mode not in ("embed", "frozen"):
            raise ValueError(f"TextFrontend mode must be 'embed' or 'frozen', got {mode!r}")
        self.mode = mode
        self.frozen_arch = frozen_arch
        self.mask_id = vocab_size                                   # input-side MASK
        self.upsample = int(upsample)

        if mode == "embed":
            self.embedding = nn.Embedding(vocab_size + 1, model_size)   # +1 for MASK
            self.proj = nn.Sequential(
                nn.Linear(model_size, model_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            return

        # mode == "frozen": a FROZEN char/byte-level encoder + trainable Linear (mirrors audio).
        # Two archs, both tokenization-free so 1 char = 1 position (keeps char-CTC alignment):
        #   canine: Unicode code points; MASK -> a private-use code point.
        #   byt5  : a stronger byte-level encoder (paper lesson: stronger frozen text encoder ->
        #           bigger UML gains). Input id = utf8_byte + 3 (past pad/eos/unk); MASK -> unk(2).
        import string as _string
        chars = chars or (_string.ascii_lowercase + _string.digits + " ")
        assert len(chars) == vocab_size, f"chars({len(chars)}) != vocab_size({vocab_size})"

        if frozen_arch == "canine":
            from transformers import CanineModel  # local import (heavy dep)
            self.encoder = CanineModel.from_pretrained(frozen_model or self.CANINE_MODEL)
            enc_dim = self.encoder.config.hidden_size
            lut = [ord(c) for c in chars] + [0xE003]          # MASK -> private-use code point
        elif frozen_arch == "byt5":
            from transformers import T5EncoderModel  # local import (heavy dep)
            self.encoder = T5EncoderModel.from_pretrained(frozen_model or self.BYT5_MODEL)
            cfg = self.encoder.config
            enc_dim = getattr(cfg, "hidden_size", None) or cfg.d_model
            lut = [ord(c) + 3 for c in chars] + [2]           # all our chars are ASCII; MASK -> unk
        else:
            raise ValueError(f"frozen_arch must be 'canine' or 'byt5', got {frozen_arch!r}")

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.projection = nn.Linear(enc_dim, model_size)
        self.register_buffer("cp_lut", torch.tensor(lut, dtype=torch.long), persistent=False)

    def forward(self, token_ids: torch.Tensor, token_lengths: torch.Tensor):
        if self.mode == "embed":
            x = self.embedding(token_ids)        # (B, T, D)
            x = self.proj(x)                     # (B, T, D)
            return x, token_lengths.long()

        # frozen: char ids -> encoder input ids -> frozen encoder -> projection -> upsample.
        input_ids = self.cp_lut[token_ids]                                    # (B, L)
        L = token_ids.shape[1]
        attn = (torch.arange(L, device=token_ids.device).unsqueeze(0)
                < token_lengths.to(token_ids.device).unsqueeze(1)).long()     # (B, L)
        with torch.no_grad():
            hidden = self.encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
        x = self.projection(hidden)                                           # (B, L, D)
        if self.upsample > 1:                                                 # match EMG frame rate + CTC margin
            x = x.repeat_interleave(self.upsample, dim=1)                     # (B, L*U, D)
        out_lengths = token_lengths.long() * self.upsample
        return x, out_lengths


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
        conv_strides=(2, 2, 2),
        num_private_layers: int = 0,
        second_branch: str = "audio",
        aux_branches: tuple[str, ...] | None = None,
        audio_frontend_mode: str = "full",
        text_frontend: str = "embed",
        text_frozen_model: str | None = None,
        text_frozen_arch: str = "canine",
        text_upsample: int = 3,
    ):
        super().__init__()
        # ``aux_branches`` (new, multi-auxiliary) takes precedence; ``second_branch`` is the
        # single-auxiliary back-compat path. Each aux gets its OWN frontend + CTC head; all feed
        # the SAME shared transformer. This operationalizes the paper's compounding Fisher info
        # (audio + text auxiliaries together).
        if aux_branches is None:
            aux_branches = (second_branch,)
        aux_branches = tuple(aux_branches)
        for b in aux_branches:
            if b not in ("audio", "text"):
                raise ValueError(f"aux branch must be 'audio' or 'text', got {b!r}")
        if len(set(aux_branches)) != len(aux_branches):
            raise ValueError(f"duplicate aux branch in {aux_branches}")
        self.aux_branches = aux_branches
        self.second_branch = aux_branches[0]  # back-compat: first aux

        self.emg_encoder = GaddyRawEMGEncoder(
            model_size=model_size, num_layers=num_layers, dropout=dropout,
            conv_strides=conv_strides, num_private_layers=num_private_layers,
        )
        if "text" in aux_branches:
            self.text_frontend = TextFrontend(
                vocab_size=vocab_size, model_size=model_size, dropout=dropout,
                mode=text_frontend, frozen_model=text_frozen_model,
                frozen_arch=text_frozen_arch, upsample=text_upsample,
            )
        if "audio" in aux_branches:
            self.audio_frontend = AudioFrontend(model_size=model_size, mode=audio_frontend_mode)

        self.share_ctc_head = bool(share_ctc_head)
        if self.share_ctc_head and len(aux_branches) > 1:
            raise ValueError("share_ctc_head is only supported with a single aux branch")
        if self.share_ctc_head:
            head = CTCHead(model_size=model_size, vocab_size=vocab_size)
            self.emg_ctc_head = head
            setattr(self, f"{aux_branches[0]}_ctc_head", head)
        else:
            self.emg_ctc_head = CTCHead(model_size=model_size, vocab_size=vocab_size)
            for b in aux_branches:
                setattr(self, f"{b}_ctc_head", CTCHead(model_size=model_size, vocab_size=vocab_size))

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
    # Text branch — uses the SAME ``self.emg_encoder.transformer`` instance
    # ------------------------------------------------------------------

    def forward_text(
        self,
        token_ids: torch.Tensor,
        token_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, out_lengths = self.text_frontend(token_ids, token_lengths)   # (B, T', D)
        # Transformer expects (T, B, D) — same convention as GaddyRawEMGEncoder.
        x = self.emg_encoder.transformer(x.transpose(0, 1)).transpose(0, 1)
        logits = self.text_ctc_head(x)
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
