from transformers import CLIPProcessor, CLIPModel
import torch

class CLIPFeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the pretrained CLIP model and processor from HuggingFace
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_features(self, image):
        # Process the image input to tensors
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        # Extract image features without gradient tracking
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        # Normalize features to unit length (optional, often useful)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features