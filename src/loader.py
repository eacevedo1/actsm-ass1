import numpy as np
import essentia.standard as es


def load_audio(path: str, sr: int = 16000, stereo: bool = False) -> np.ndarray:
    """
    Loads an audio file using Essentia's AudioLoader or MonoLoader.

    :param path: Path to the audio file to load.
    :param sr: Target sample rate in Hz to resample the audio to. Defaults to 16000.
    :param stereo: If True, loads the audio in stereo (2 channels) using AudioLoader.
                   If False, loads as a mono signal using MonoLoader. Defaults to False.
    :return: Loaded audio signal as a numpy array.
             - Stereo: shape (n_samples, 2)
             - Mono: shape (n_samples,)
    """

    if stereo:
        audio, _, _, _, _, _ = es.AudioLoader(filename=path, sampleRate=sr)()
    else:
        audio = es.MonoLoader(filename=path, sampleRate=sr)()
    return audio
