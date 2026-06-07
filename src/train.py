import torch

from torch.utils.data import DataLoader

from transformers import AdamW

from dataset import OCRDataset

from model import (
    model,
    processor
)

from utils import get_device

TRAIN_CSV = "data/annotations/train.csv"

BATCH_SIZE = 4

EPOCHS = 10

LR = 5e-5

device = get_device()

model.to(device)

train_dataset = OCRDataset(
    TRAIN_CSV,
    processor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

optimizer = AdamW(
    model.parameters(),
    lr=LR
)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch in train_loader:

        pixel_values = batch[
            "pixel_values"
        ].to(device)

        labels = batch[
            "labels"
        ]

        labels = torch.tensor(
            labels
        ).to(device)

        outputs = model(

            pixel_values=pixel_values,

            labels=labels

        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        total_loss += loss.item()

    print(

        f"Epoch {epoch+1}"

        f" Loss : "

        f"{total_loss}"

    )

torch.save(

    model.state_dict(),

    "models/final_model/model.pth"

)

print(
    "Training Finished"
)