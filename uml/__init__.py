"""UML (unpaired modality learning) helpers for sEMGencoderJEPA.

Contents
--------
- :mod:`uml.audio_dataset` : LibriSpeech cache reader (waveform + char text_int).
- :mod:`uml.model`         : ``AudioFrontend`` (frozen Wav2Vec2 + projection) and
                             ``UMLModel`` (dual-branch model with shared
                             Transformer between EMG and audio branches).
"""
