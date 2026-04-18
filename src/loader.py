import numpy as np
import essentia.standard as es


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """
    Loads an audio file using Essentia's AudioLoader.

    :param path: Path to the audio file to load.
    :return: Loaded audio signal as a numpy array and the sample rate.
    """

    loader = es.AudioLoader(filename=path)
    audio, sample_rate, number_channels, md5, bit_rate, codec = loader()

    return audio, sample_rate


def convert_to_mono(x: np.ndarray) -> np.ndarray:
    """
    Converts a stereo audio signal to mono using Essentia's MonoMixer.

    :param x: Stereo audio array of shape (samples, channels).
    :return: Mono audio array of shape (samples,).
    """
    y = es.MonoMixer().compute(x, x.shape[1])

    return y


def resample(x: np.ndarray, src_sr: int, dst_sr: int) -> tuple[np.ndarray, int]:
    """
    Resamples a mono audio signal to a target sample rate.

    :param x: Mono audio array of shape (samples,).
    :param src_sr: Source sample rate in Hz.
    :param dst_sr: Target sample rate in Hz.
    :return: Resampled audio array and the target sample rate.
    """
    assert x.ndim == 1, f"Expected mono audio (samples,), got shape {x.shape}"

    y = es.Resample(
        inputSampleRate=float(src_sr), outputSampleRate=float(dst_sr), quality=0
    ).compute(x)

    return y, dst_sr
