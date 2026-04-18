import essentia.standard as es


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
