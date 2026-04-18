import numpy as np
import essentia.standard as es


def bpm_extractor(x: np.ndarray) -> tuple[float, float]:
    """
    BPM extractor.

    :param x: Audio signal as a numpy array
    :return: Tuple of (bpm, confidence)
        - bpm: Tempo of the audio in beats per minute
        - confidence: Confidence of the BPM estimation (0-1)
    """

    bpm, _, confidence, _, _ = es.RhythmExtractor2013().compute(x)

    return bpm, confidence


def key_extractor(x: np.ndarray) -> tuple[str, str, float]:
    """
    Key extractor.

    :param x: Audio signal as a numpy array
    :return: Tuple of (key, scale, confidence)
        - key: Musical key of the audio (e.g. 'C', 'G#')
        - scale: Scale of the audio (e.g. 'major', 'minor')
        - confidence: Confidence of the key estimation (0-1)
    """

    return es.KeyExtractor().compute(x)


def loudnessEBUR_extractor(x: np.ndarray) -> float:
    """
    LoudnessEBUR128 extractor.

    :param x: Stereo Audio signal as a numpy array
    :return: Float of integrated loudness (overall) (LUFS)
    """

    assert x.ndim == 2 and x.shape[1] == 2, (
        f"Expected stereo audio (samples, 2), got shape {x.shape}"
    )

    _, _, integratedLoudness, _ = es.LoudnessEBUR128().compute(x)

    return integratedLoudness
