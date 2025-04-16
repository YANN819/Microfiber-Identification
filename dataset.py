import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from PIL import Image, ImageChops
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import Dataset



class SpectrumDataset1D(Dataset):
    """
    Dataset for 1D Neural Network (Spectrum data)
    :param csv_file: Path to the CSV file containing spectrum data.
    :param in_channel: Number of input channels.
    :param transform: Optional data transformations.
    :param max_length: Maximum length of the spectrum data.
    """

    def __init__(self, csv_file, in_channel, transform=None, max_length=None):
        self.data_frame = pd.read_csv(csv_file)
        self.in_channel = in_channel
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.data_frame)

    
    def __getitem__(self, idx):
        spectrum_data = self.data_frame.iloc[idx, 1:].values.astype(float)
        label = int(self.data_frame.iloc[idx, 0])

        if self.max_length:
            spectrum_data = self.pad_or_trim(spectrum_data, self.max_length)

        if self.in_channel == 1:
            spectrum_data = spectrum_data[None, :] 
        elif spectrum_data.shape[0] != self.in_channel:
            raise ValueError(f"Expected {self.in_channel} channels, but got {spectrum_data.shape[0]} channels.")

        spectrum_data = torch.from_numpy(spectrum_data).float()
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            spectrum_data = self.transform(spectrum_data)

        return spectrum_data, label

    def pad_or_trim(self, spectrum_data, length):
        current_length = len(spectrum_data)
        if current_length < length:
            
            spectrum_data = F.pad(spectrum_data, (0, length - current_length), value=0)
        elif current_length > length:
            
            spectrum_data = spectrum_data[:length]
        return spectrum_data


class SpectrumDataset2D(Dataset):
    """
    Dataset for 2D Convolutional Neural Network (Spectrum image data)
    :param root_dir: Path to the directory containing spectrum image data.
    :param transform: Optional data transformations.
    :param im_size: Size of input image (height, width)
    :param resample: Resampling method for image resizing
    """

    def __init__(
        self,
        root_dir,
        transform = None,
        resample = Image.Resampling.LANCZOS,
        im_size: Tuple[int, int] = (480, 480)
    ):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.transform = transform
        self.im_size = (im_size[1], im_size[0])
        self.resample = resample

    def crop_bbox(self, image: Image.Image):

        bg = Image.new(image.mode, image.size, "white")
        diff = ImageChops.difference(image, bg)
        image = image.crop(diff.getbbox())
        return image

    def __len__(self):
        total_samples = sum(len(files) for _, _, files in os.walk(self.root_dir))
        return total_samples

    def __getitem__(self, idx):
        label = None
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            files = os.listdir(class_dir)
            if idx < len(files):
                label = int(class_name)
                img_name = os.path.join(class_dir, files[idx])
                break
            else:
                idx -= len(files)

        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        image = self.crop_bbox(image)
        image = image.resize(self.im_size, resample=self.resample)

        if self.transform:
            image = self.transform(image)

        arr = np.asarray(image, dtype=np.float32)
        spectrum_data = torch.from_numpy(arr).unsqueeze(0)
        return spectrum_data, label


if __name__ == "__main__":

    root_dir = r'D:\Anaconda3\pythonwork\ramanspec_file\data_training\cnn\spectra_images'

    spectrum_image_dataset = SpectrumDataset2D(root_dir, transform=transforms.Grayscale())

    sample = spectrum_image_dataset[1200]
    image, label = sample
    
    print(type(label))
    print("Image shape:", image.shape)
    print("Label:", label)

    image_np = image.squeeze().numpy()
    plt.imshow(image_np, cmap='gray')
    plt.axis()
    plt.show()

