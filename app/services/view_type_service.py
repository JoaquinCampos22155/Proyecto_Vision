from dataclasses import dataclass

from PIL import Image, ImageFilter


VIEW_TYPE_NOTE = "Heuristic MVP estimate"
MAIN_VIEW_TYPES = {"main_front", "main_back", "partial_view"}
DETAIL_VIEW_TYPES = {"detail_collar", "detail_logo", "detail_label", "detail_fabric"}


@dataclass(frozen=True)
class ViewTypeResult:
    image_index: int
    view_type: str
    confidence: float
    note: str = VIEW_TYPE_NOTE
    filename: str = ""


class ViewTypeService:
    def estimate(self, image: Image.Image, image_index: int, filename: str = "") -> ViewTypeResult:
        name = filename.lower()
        # Filename metadata is only a weak view-type hint from controlled ingest/debug flows.
        # It must never be treated as ground-truth evidence that a product matches or mismatches.
        if "back" in name or "espalda" in name:
            return ViewTypeResult(image_index, "main_back", 0.72, filename=filename)
        if "front" in name or "frente" in name:
            return ViewTypeResult(image_index, "main_front", 0.72, filename=filename)
        if "collar" in name or "cuello" in name:
            return ViewTypeResult(image_index, "detail_collar", 0.74, filename=filename)
        if "logo" in name:
            return ViewTypeResult(image_index, "detail_logo", 0.74, filename=filename)
        if "label" in name or "tag" in name or "etiqueta" in name:
            return ViewTypeResult(image_index, "detail_label", 0.74, filename=filename)
        if "fabric" in name or "tela" in name:
            return ViewTypeResult(image_index, "detail_fabric", 0.70, filename=filename)

        metrics = self._image_metrics(image)
        foreground_ratio = metrics["foreground_ratio"]
        edge_density = metrics["edge_density"]
        center_foreground_ratio = metrics["center_foreground_ratio"]

        # These rules are intentionally simple and documented so they can be replaced by a trained
        # view classifier later. They separate full/partial product views from close-up/detail shots.
        if foreground_ratio >= 0.72 and edge_density >= 0.17:
            return ViewTypeResult(image_index, "detail_fabric", 0.58, filename=filename)

        if foreground_ratio >= 0.62 and edge_density >= 0.11:
            return ViewTypeResult(image_index, "detail_logo", 0.56, filename=filename)

        if foreground_ratio >= 0.52 and center_foreground_ratio >= 0.50:
            return ViewTypeResult(image_index, "partial_view", 0.60, filename=filename)

        if foreground_ratio >= 0.34:
            return ViewTypeResult(image_index, "main_front", 0.55, filename=filename)

        if foreground_ratio >= 0.20:
            return ViewTypeResult(image_index, "partial_view", 0.48, filename=filename)

        return ViewTypeResult(image_index, "unknown", 0.35, filename=filename)

    @staticmethod
    def _image_metrics(image: Image.Image) -> dict[str, float]:
        import numpy as np

        rgb = image.convert("RGB").resize((128, 128))
        array = np.asarray(rgb, dtype=np.int16)
        corners = np.concatenate(
            [
                array[:16, :16].reshape(-1, 3),
                array[:16, -16:].reshape(-1, 3),
                array[-16:, :16].reshape(-1, 3),
                array[-16:, -16:].reshape(-1, 3),
            ],
            axis=0,
        )
        background = np.median(corners, axis=0)
        distances = np.linalg.norm(array - background, axis=2)
        foreground_mask = distances > 34
        center_mask = foreground_mask[32:96, 32:96]

        edges = rgb.convert("L").filter(ImageFilter.FIND_EDGES)
        edge_array = np.asarray(edges, dtype=np.uint8)
        edge_density = float((edge_array > 36).mean())

        return {
            "foreground_ratio": float(foreground_mask.mean()),
            "center_foreground_ratio": float(center_mask.mean()),
            "edge_density": edge_density,
        }
