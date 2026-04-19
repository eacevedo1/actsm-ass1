import essentia.standard as es
import laion_clap


def load_discogs400Effnet_models() -> tuple[
    es.TensorflowPredictEffnetDiscogs, es.TensorflowPredict2D
]:
    """
    Loads Discogs-EffNet and Genre Discogs400 models.

    :return: Tuple of (effnet_model, genre_model) ready for inference.
    """
    effnet = es.TensorflowPredictEffnetDiscogs(
        graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1"
    )
    genre = es.TensorflowPredict2D(
        graphFilename="models/genre_discogs400-discogs-effnet-1.pb",
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",
    )
    return effnet, genre


def load_voiceinstrumental_model() -> es.TensorflowPredict2D:
    """
    Loads the voice/instrumental classifier model.

    :return: TensorflowPredict2D model ready for inference.
    """
    return es.TensorflowPredict2D(
        graphFilename="models/voice_instrumental-discogs-effnet-1.pb",
        output="model/Softmax",
    )


def load_danceability_model() -> es.TensorflowPredict2D:
    """
    Loads the danceability classifier model.

    :return: TensorflowPredict2D model ready for inference.
    """
    return es.TensorflowPredict2D(
        graphFilename="models/danceability-discogs-effnet-1.pb",
        output="model/Softmax",
    )


def load_clap_model() -> laion_clap.CLAP_Module:
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt("models/music_speech_epoch_15_esc_89.25.pt")
    return model
