from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

MODEL_NAME = "microsoft/trocr-base-handwritten"

processor = TrOCRProcessor.from_pretrained(
    MODEL_NAME
)

model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_NAME
)

model.config.decoder_start_token_id = (
    processor.tokenizer.cls_token_id
)

model.config.pad_token_id = (
    processor.tokenizer.pad_token_id
)

model.config.eos_token_id = (
    processor.tokenizer.sep_token_id
)