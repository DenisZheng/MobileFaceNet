import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_DIR, pickle_file, NORM_MEAN, NORM_STD

# Data augmentation and normalization for training
# Just normalization for validation
# Normalization constants come from config.py (NORM_PRESET) so training,
# LFW evaluation and deployment preprocessing can never drift apart.
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ]),
}


class ArcFaceDataset(Dataset):
    def __init__(self, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.split = split
        self.samples = data

        self.transformer = data_transforms['train']

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = sample['img']
        filename = os.path.join(IMG_DIR, filename)
        img = Image.open(filename).convert('RGB')
        img = self.transformer(img)

        label = sample['label']

        return img, label

    def __len__(self):
        return len(self.samples)
