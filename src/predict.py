from PIL import Image

import torch

def predict(
        image_path,
        model,
        processor,
        device
):

    image = Image.open(
        image_path
    ).convert("RGB")

    pixel_values = processor(
        image,
        return_tensors="pt"
    ).pixel_values

    pixel_values = pixel_values.to(
        device
    )

    generated_ids = model.generate(
        pixel_values
    )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return generated_text