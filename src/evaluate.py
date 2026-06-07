from jiwer import wer
from jiwer import cer

def calculate_metrics(
        predictions,
        references
):

    word_error = wer(
        references,
        predictions
    )

    char_error = cer(
        references,
        predictions
    )

    return {

        "WER": word_error,
        "CER": char_error

    }