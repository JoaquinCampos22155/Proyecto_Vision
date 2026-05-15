from io import BytesIO

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.services.validation_service_types import LoadedImage


class ImageLoadingError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ImageLoader:
    async def load(self, upload: UploadFile, image_index: int) -> LoadedImage:
        if not upload.content_type or not upload.content_type.startswith("image/"):
            raise ImageLoadingError(
                f"File at index {image_index} is not an image. Received content type: {upload.content_type!r}."
            )

        raw = await upload.read()
        if not raw:
            raise ImageLoadingError(f"File at index {image_index} is empty.")

        try:
            image = Image.open(BytesIO(raw))
            image.verify()
            image = Image.open(BytesIO(raw)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ImageLoadingError(f"File at index {image_index} could not be decoded as an image.") from exc

        return LoadedImage(index=image_index, filename=upload.filename or f"image_{image_index}", image=image)
