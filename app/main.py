from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.schemas.response_schemas import ValidationResponse
from app.services.validation_service import ProductImageValidationService, ValidationError


STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Visual Product Validation API",
    version="0.1.0",
    description="MVP API for detecting visual inconsistencies across product listing images.",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(ValidationError)
async def validation_error_handler(_, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": exc.message})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/validate-product-images", response_model=ValidationResponse)
async def validate_product_images(
    images: Annotated[list[UploadFile], File(description="Between 6 and 10 product images")],
    product_id: str | None = Form(default=None, description="Optional product identifier"),
) -> ValidationResponse:
    settings = get_settings()
    if len(images) < settings.min_images:
        raise HTTPException(
            status_code=400,
            detail=f"At least {settings.min_images} images are required.",
        )
    if len(images) > settings.max_images:
        raise HTTPException(
            status_code=400,
            detail=f"No more than {settings.max_images} images are allowed.",
        )

    service = ProductImageValidationService(settings=settings)
    return await service.validate(product_id=product_id, uploads=images)
