# Visual Product Validation API

MVP en FastAPI para validar si entre 6 y 10 imagenes de una publicacion parecen ser compatibles con el mismo producto. El sistema usa embeddings visuales, tipos de toma heurísticos y color dominante aproximado para evitar castigar variaciones normales de ecommerce como frente, espalda, close-ups, cuello, logo o etiqueta.

## Stack

- Python + FastAPI
- PyTorch / torchvision para embeddings con ResNet preentrenada
- Ultralytics YOLO preparado para recorte opcional de la prenda
- Pillow para carga, recorte y color dominante
- Pytest para pruebas basicas

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> En Linux/macOS usa `source .venv/bin/activate`.

La primera llamada al endpoint puede descargar los pesos de `yolov8n.pt` y de ResNet desde sus fuentes oficiales.

## Ejecutar la API

```bash
uvicorn app.main:app --reload
```

Luego abre:

- Healthcheck: http://127.0.0.1:8000/health
- Documentacion Swagger: http://127.0.0.1:8000/docs

## Probar el endpoint

```bash
curl -X POST "http://127.0.0.1:8000/validate-product-images" \
  -F "product_id=sku-123" \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "images=@image3.jpg" \
  -F "images=@image4.jpg" \
  -F "images=@image5.jpg" \
  -F "images=@image6.jpg"
```

El endpoint acepta entre 6 y 10 archivos en `multipart/form-data`.

## Testing with the simple UI

Corre la API en el puerto 8081:

```bash
py -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8081
```

Abre la UI en el navegador:

http://127.0.0.1:8081/

Para probar:

1. Ingresa un `product_id` si quieres asociar la prueba a un identificador.
2. Selecciona entre 6 y 10 imagenes del mismo producto.
3. Revisa el preview y el contador de imagenes.
4. Presiona `Validate Images`.
5. Revisa el resumen visual y el JSON completo que devuelve la API.

## Visual similarity vs product compatibility

La similitud visual pura pregunta si todas las imagenes se parecen entre si. Eso falla en publicaciones reales: una foto frontal completa, una foto de espalda, un close-up del cuello y una etiqueta pueden tener baja similitud global aunque pertenezcan a la misma camisa.

Este MVP intenta medir compatibilidad de producto:

- Clasifica cada imagen con un tipo de toma heuristico: `main_front`, `main_back`, `partial_view`, `detail_collar`, `detail_logo`, `detail_label`, `detail_fabric` o `unknown`.
- Da mas peso a comparaciones entre vistas principales.
- Usa imagenes de detalle como evidencia secundaria, no como razon automatica de rechazo.
- Agrega consistencia de color dominante como senal auxiliar.
- Mantiene la matriz completa de similitud para debug.

Las imagenes de detalle no se comparan con la misma exigencia que las principales porque suelen mostrar zoom, textura, etiqueta, cuello o logo. Una imagen de detalle solo se marca como `flagged` si tambien tiene color muy distinto o no aporta soporte visual minimo.

## Respuesta

La API devuelve:

- `status`: `consistent`, `needs_review` o `inconsistent`.
- `scores.raw_consistency_score`: promedio simple de similitudes coseno entre pares, excluyendo la diagonal.
- `scores.main_view_score`: score entre imagenes `main_front`, `main_back` y `partial_view`; puede ser `null`.
- `scores.detail_support_score`: soporte auxiliar de imagenes de detalle.
- `scores.color_consistency_score`: compatibilidad de color dominante aproximado.
- `scores.robust_consistency_score`: score compuesto usado para decidir el `status`.
- `view_types`: tipo de toma estimado por imagen.
- `flagged_images`: imagenes que requieren revision contextualizada, con severidad y accion recomendada.
- `dominant_colors`: color dominante estimado sobre el recorte usado para embeddings.
- `garment_type_estimates`: atributo auxiliar basado en YOLO, no usado como decision principal.
- `image_debug`: informacion de crop/deteccion por imagen, incluyendo `crop_strategy` y fallback.
- `crop_debug_urls`: URLs de los crops guardados temporalmente en `app/static/debug_crops/`.
- `condition_estimate`: placeholder estructurado para futura estimacion de estado nuevo/usado.
- `pairwise_similarity_matrix`: matriz completa de similitud coseno.
- `note`: advertencia de calibracion del MVP.

## Configuracion

Los thresholds estan centralizados en `app/config.py` y tambien pueden ajustarse con variables de entorno:

```bash
CONSISTENT_THRESHOLD=0.70
REVIEW_THRESHOLD=0.50
MAIN_VIEW_WEIGHT=0.65
COLOR_CONSISTENCY_WEIGHT=0.25
DETAIL_SUPPORT_WEIGHT=0.10
USE_YOLO_CROPS=false
DEFAULT_CROP_STRATEGY=full_image
YOLO_MODEL_PATH=yolov8n.pt
RESNET_MODEL_NAME=resnet50
DEVICE=auto
```

Reglas actuales:

- `robust_consistency_score >= 0.70`: `consistent`
- `0.50 <= robust_consistency_score < 0.70`: `needs_review`
- `robust_consistency_score < 0.50`: `inconsistent`

Tambien puede devolver `needs_review` si hay una imagen dudosa pero no suficiente evidencia para decir que es otro producto. Puede devolver `inconsistent` si hay una imagen principal claramente incompatible o un mismatch fuerte de color entre imagenes principales.

Por defecto `USE_YOLO_CROPS=false`, asi que el embedding principal usa la imagen completa. Si activas YOLO, solo se aceptan detecciones con confianza alta, clases permitidas, bbox suficientemente grande y ubicacion razonable; si no, se usa fallback.

## Arquitectura

```text
app/
  main.py
  config.py
  static/
    index.html
    styles.css
    script.js
    debug_crops/
  services/
    image_loader.py
    garment_detector.py
    embedding_service.py
    similarity_service.py
    scoring_service.py
    view_type_service.py
    color_service.py
    condition_service.py
    validation_service.py
  schemas/
    response_schemas.py
  utils/
    image_utils.py
tests/
  test_similarity_service.py
  test_scoring_service.py
  test_endpoint_product_id.py
```

## Notas de extension

- No hay entrenamiento de modelos en este MVP.
- No hay base de datos.
- Incluye una UI estatica simple para pruebas locales; no usa frameworks frontend.
- `app/services/garment_detector.py` esta preparado para reemplazar `yolov8n.pt` por un YOLO fine-tuned para prendas cuando existan datos etiquetados.
- La clasificacion de tipo de toma es heuristica y debe reemplazarse o calibrarse con datos reales.
- El color dominante intenta ignorar fondo con una heuristica simple; no es segmentacion real.
- La decision principal viene de embeddings, compatibilidad por tipo de toma y color auxiliar.

## Dataset evaluation workflow

El dataset en `images/` puede auditarse, dividirse y evaluarse contra el pipeline actual sin modificar imagenes ni `labels.json`.

Auditar labels y estructura:

```bash
python scripts/audit_dataset.py --dataset-dir images --output-dir reports
```

Crear splits estratificados por `case_type`:

```bash
python scripts/create_dataset_splits.py --dataset-dir images --output-dir reports --seed 42
```

Evaluar todos los casos validos en modo directo:

```bash
py scripts/evaluate_dataset.py --dataset-dir images --output-dir reports --split all --mode direct
```

Evaluar un split especifico en modo directo:

```bash
py scripts/evaluate_dataset.py --dataset-dir images --output-dir reports --split validation --mode direct
```

Tambien existe un wrapper corto:

```bash
py evaluate.py --dataset-dir images --output-dir reports --split validation --mode direct
```

Los splits soportados son `calibration`, `validation`, `test` y `real_world`. `validation` se usa para debugging y regresion; `test` debe tratarse como medicion final no usada para ajustar reglas a mano; `real_world` queda para muestras de QA manual o datos de produccion.

Para guardar un run versionado:

```bash
py scripts/evaluate_dataset.py --dataset-dir images --output-dir reports --split validation --mode direct --run-name candidate_validation
```

Tambien puedes evaluar usando la API HTTP. Primero corre la API:

```bash
py -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8081
```

Luego corre la evaluacion por HTTP:

```bash
py scripts/evaluate_dataset.py --dataset-dir images --output-dir reports --split all --mode http --api-url http://127.0.0.1:8081
```

El modo `auto` intenta usar el servicio interno y, si falla el import y existe `--api-url`, usa HTTP como fallback.

Archivos generados:

- `reports/dataset_audit_summary.json`: conteo general de carpetas, labels validos e issues.
- `reports/dataset_audit_issues.csv`: detalle de problemas por carpeta.
- `reports/dataset_splits.json`: asignacion de casos a `calibration`, `validation` y `test`.
- `reports/dataset_split_summary.csv`: conteos por split y `case_type`.
- `reports/evaluation_results.csv`: resultado resumido por caso.
- `reports/evaluation_results.json`: respuesta completa del sistema por caso.
- `reports/evaluation_summary.json`: metricas generales, por `case_type` y matriz de confusion.
- `reports/evaluation_errors.csv`: casos donde falla el status aceptable o los flagged esperados.
- `reports/error_cases/`: JSON por cada caso que requiere revision manual.
- `reports/regression_report.json`: delta contra el baseline congelado, con warnings por categoria protegida.
- `reports/no_overfitting_report.json`: resumen de riesgos de overfitting y regresiones.

Baseline congelado:

- `docs/evaluation_baselines/baseline_v0_3_post_fix.md`
- `docs/evaluation_baselines/baseline_v0_3_post_fix.json`

Protocolo de evaluacion:

- `docs/evaluation_protocol.md`

Como interpretar las metricas:

- `exact_match_accuracy`: porcentaje donde `predicted_status` coincide exactamente con `expected_status`.
- `acceptable_match_accuracy`: porcentaje donde `predicted_status` esta dentro de `acceptable_status`; si no existe, usa `expected_status`.
- `flagged_exact_match_accuracy`: porcentaje donde los filenames flagged predichos coinciden exactamente con los esperados.
- `flagged_partial_match_accuracy`: porcentaje donde al menos una imagen esperada como flagged fue detectada; si ambos sets estan vacios, cuenta como correcto.
- `confusion_matrix`: cruza `expected_status` contra `predicted_status`.
- `by_case_type`: muestra las mismas metricas agrupadas por tipo de caso.

El servicio `app/services/invalid_image_service.py` contiene checks iniciales para `too_dark`, `blurry_image`, `low_resolution` y `duplicate_image`. Las senales de calidad se separan de los mismatches de producto: una advertencia de calidad moderada no deberia cambiar el status cuando la evidencia visual del producto es fuerte.

## Tests

```bash
pytest
```
