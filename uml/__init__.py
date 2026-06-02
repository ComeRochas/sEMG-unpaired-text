"""UML (unpaired multimodal learning) helpers — kept as a reference template.

The audio branch is retained so the shared-Transformer recipe can be extended to
a future text branch (and to >2 modalities, e.g. EMG+text+audio). Phase-2's
unpaired-text branch will mirror these two modules.

Contents
--------
- :mod:`uml.audio_dataset` : audio cache reader (waveform + char text_int).
- :mod:`uml.model`         : ``AudioFrontend`` (frozen Wav2Vec2 + projection) and
                             ``UMLModel`` (dual-branch model with shared
                             Transformer between EMG and audio branches).
"""
