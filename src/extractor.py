import numpy as np
import essentia.standard as es
import laion_clap


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


def key_extractor(x: np.ndarray, profileType: str = "bgate") -> tuple[str, str, float]:
    assert x.ndim == 1, f"Expected mono audio (samples,), got shape {x.shape}"

    return es.KeyExtractor(profileType=profileType).compute(x)


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


def effnet_classifier(
    embeddings: np.ndarray, model: es.TensorflowPredict2D
) -> np.ndarray:
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


def clap_audio_extractor(
    audio: np.ndarray, model: laion_clap.CLAP_Module
) -> np.ndarray:
    """
    Extracts CLAP audio embedding from a mono 48kHz audio array.

    :param audio: Mono audio array of shape (samples,) at 48kHz.
    :param model: Pre-loaded CLAP_Module.
    :return: Audio embedding of shape (512,).
    """
    assert audio.ndim == 1, f"Expected mono audio (samples,), got shape {audio.shape}"
    x = audio.reshape(1, -1)
    embedding = model.get_audio_embedding_from_data(x=x, use_tensor=False)
    return embedding.squeeze(0)


def clap_text_extractor(texts: list[str], model: laion_clap.CLAP_Module) -> np.ndarray:
    """
    Extracts CLAP text embeddings from a list of strings.

    :param texts: List of text queries.
    :param model: Pre-loaded CLAP_Module.
    :return: Text embeddings of shape (n_texts, 512).
    """
    return model.get_text_embedding(texts, use_tensor=False)
