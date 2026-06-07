import pandas as pd

from torch.utils.data import Dataset

from preprocess import load_image

class OCRDataset(Dataset):

    def __init__(
            self,
            csv_file,
            processor,
            max_target_length=512
    ):

        self.data = pd.read_csv(csv_file)

        self.processor = processor

        self.max_target_length = max_target_length

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        image_path = self.data.iloc[idx]["image"]

        text = self.data.iloc[idx]["text"]

        image = load_image(image_path)

        pixel_values = self.processor(
            image,
            return_tensors="pt"
        ).pixel_values

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        ).input_ids

        labels = [
            label if label != self.processor.tokenizer.pad_token_id
            else -100
            for label in labels
        ]

        return {

            "pixel_values":
            pixel_values.squeeze(),

            "labels":
            labels

        }