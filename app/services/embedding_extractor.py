import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


class EmbeddingExtractor:
    """
    Extrae embeddings visuales usando ResNet50 preentrenada.

    La última capa de clasificación se elimina.
    El resultado es un vector numérico que representa visualmente la imagen.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        weights = ResNet50_Weights.DEFAULT

        base_model = resnet50(weights=weights)

        self.model = torch.nn.Sequential(
            *list(base_model.children())[:-1]
        )

        self.model.to(self.device)
        self.model.eval()

        self.preprocess = weights.transforms()

    @torch.no_grad()
    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Recibe una imagen PIL y devuelve un embedding normalizado.
        """

        image = image.convert("RGB")

        tensor = self.preprocess(image)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        embedding = self.model(tensor)
        embedding = embedding.flatten()
        embedding = embedding.cpu().numpy()

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return embedding

        return embedding / norm