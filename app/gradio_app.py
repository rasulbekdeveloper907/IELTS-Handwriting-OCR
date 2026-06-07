import gradio as gr

import torch

from model import (
    model,
    processor
)

from predict import predict

from utils import get_device

device = get_device()

model.load_state_dict(

    torch.load(

        "models/final_model/model.pth",

        map_location=device

    )

)

model.to(device)

model.eval()

def ocr(image):

    image.save(
        "temp.jpg"
    )

    text = predict(

        "temp.jpg",

        model,

        processor,

        device

    )

    return text

demo = gr.Interface(

    fn=ocr,

    inputs=gr.Image(
        type="pil"
    ),

    outputs="text",

    title="IELTS Handwriting OCR"

)

demo.launch()