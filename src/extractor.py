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
    assert x.ndim == 1, f"Expected mono audio (samples,), got shape {x.shape}"

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
    assert x.ndim == 1, f"Expected mono audio (samples,), got shape {x.shape}"

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


def discogs400Effnet_extractor(
    x: np.ndarray,
    effnet_model: es.TensorflowPredictEffnetDiscogs,
    genre_model: es.TensorflowPredict2D,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts Discogs-EffNet embeddings and Genre Discogs400 predictions for a single track.

    :param x: Mono audio array of shape (samples,) at 16kHz.
    :param effnet_model: Pre-loaded TensorflowPredictEffnetDiscogs model.
    :param genre_model: Pre-loaded TensorflowPredict2D genre model.
    :return: Tuple of (embeddings, predictions)
        - embeddings: EffNet frame embeddings of shape (n_frames, 1280)
        - predictions: Genre activations of shape (n_frames, 400)
    """
    assert x.ndim == 1, f"Expected mono audio (samples,), got shape {x.shape}"
    embeddings = effnet_model(x)
    predictions = genre_model(embeddings)
    return embeddings, predictions


def effnet_classifier(embeddings: np.ndarray, model: es.TensorflowPredict2D) -> np.ndarray:
    """
    Runs a Discogs-EffNet-based classifier on frame embeddings.

    :param embeddings: EffNet frame embeddings of shape (n_frames, 1280).
    :param model: Pre-loaded TensorflowPredict2D classifier model.
    :return: Softmax predictions of shape (n_frames, n_classes).
    """
    assert embeddings.ndim == 2 and embeddings.shape[1] == 1280, (
        f"Expected embeddings (n_frames, 1280), got shape {embeddings.shape}"
    )
    return model(embeddings)
