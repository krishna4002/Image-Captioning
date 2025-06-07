import os
import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class IUXRayDataset(Dataset):
    def __init__(self, json_path, image_folder):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for entry in self.data:
            if entry['sentences']:
                caption = " ".join(entry['sentences'][0]['tokens'])
                image_path = os.path.join(self.image_folder, entry['img_paths'][0])
                samples.append((image_path, caption))
        return samples

    def _len_(self):
        return len(self.samples)

    def _getitem_(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), caption

class CocoDataset(Dataset):
    def __init__(self, json_path, image_folder):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for item in self.data:
            img = os.path.join(self.image_folder, item["file_name"])
            caption = item["caption"]
            samples.append((img, caption))
        return samples

    def _len_(self):
        return len(self.samples)

    def _getitem_(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), caption