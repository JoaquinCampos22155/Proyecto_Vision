import torch
from PIL import Image
from torch import nn
from torchvision import models

from app.config import Settings


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: nn.Module | None = None
        self._preprocess = None
        self._device = self._resolve_device(settings.device)

    @staticmethod
    def _resolve_device(configured_device: str) -> torch.device:
        if configured_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(configured_device)

    def _build_model(self) -> tuple[nn.Module, object]:
        if self.settings.resnet_model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        else:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)

        model.fc = nn.Identity()
        model.eval()
        model.to(self._device)
        return model, weights.transforms()

    @property
    def model(self) -> nn.Module:
        if self._model is None or self._preprocess is None:
            self._model, self._preprocess = self._build_model()
        return self._model

    @property
    def preprocess(self):
        if self._model is None or self._preprocess is None:
            self._model, self._preprocess = self._build_model()
        return self._preprocess

    def extract_embeddings(self, images: list[Image.Image]) -> torch.Tensor:
        tensors = [self.preprocess(image).unsqueeze(0) for image in images]
        batch = torch.cat(tensors, dim=0).to(self._device)

        with torch.inference_mode():
            embeddings = self.model(batch)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu()
