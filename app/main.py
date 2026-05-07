from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.schemas.response_schema import ValidationResponse
from app.services.product_validator import ProductValidationService
from app.utils.image_utils import load_image_from_upload_bytes


app = FastAPI(
    title="Visual Product Validation API",
    description="API de validación visual para detectar inconsistencias entre múltiples imágenes de un mismo producto.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

validator_service = ProductValidationService()


@app.get("/")
def root():
    return {
        "message": "Visual Product Validation API is running",
        "version": "1.0.0",
        "main_endpoint": "POST /validate-product",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "visual-validation-api",
    }


@app.post("/validate-product", response_model=ValidationResponse)
async def validate_product(
    product_id: str = Form(...),
    images: List[UploadFile] = File(...),
):
    """
    Recibe entre 6 y 10 imágenes de un mismo producto y devuelve un análisis de consistencia visual.
    """

    if len(images) < 6:
        raise HTTPException(
            status_code=400,
            detail="You must upload at least 6 images.",
        )

    if len(images) > 10:
        raise HTTPException(
            status_code=400,
            detail="You can upload a maximum of 10 images.",
        )

    loaded_images = []

    for image_file in images:
        if not image_file.filename:
            raise HTTPException(
                status_code=400,
                detail="Every uploaded image must have a filename.",
            )

        allowed_extensions = (".jpg", ".jpeg", ".png", ".webp")
        filename_lower = image_file.filename.lower()

        if not filename_lower.endswith(allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format for {image_file.filename}. Allowed formats: jpg, jpeg, png, webp.",
            )

        try:
            image_bytes = await image_file.read()

            image = load_image_from_upload_bytes(
                image_bytes=image_bytes,
                filename=image_file.filename,
            )

            loaded_images.append(
                {
                    "filename": image_file.filename,
                    "image": image,
                }
            )

        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            )

    try:
        result = validator_service.validate(
            product_id=product_id,
            images=loaded_images,
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Internal validation error: {str(error)}",
        )