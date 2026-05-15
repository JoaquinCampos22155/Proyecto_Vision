const form = document.querySelector("#validation-form");
const imageInput = document.querySelector("#images");
const productIdInput = document.querySelector("#product-id");
const imageCount = document.querySelector("#image-count");
const previewGrid = document.querySelector("#preview-grid");
const message = document.querySelector("#message");
const validateButton = document.querySelector("#validate-button");
const resultPanel = document.querySelector("#result-panel");
const statusBadge = document.querySelector("#status-badge");
const productIdSent = document.querySelector("#product-id-sent");
const productIdReturned = document.querySelector("#product-id-returned");
const summaryGrid = document.querySelector("#summary-grid");
const resultNote = document.querySelector("#result-note");
const flaggedImages = document.querySelector("#flagged-images");
const dominantColors = document.querySelector("#dominant-colors");
const conditionEstimate = document.querySelector("#condition-estimate");
const matrixWrap = document.querySelector("#matrix-wrap");
const cropDebugGrid = document.querySelector("#crop-debug-grid");
const viewTypesWrap = document.querySelector("#view-types-wrap");
const imageDebugWrap = document.querySelector("#image-debug-wrap");
const jsonOutput = document.querySelector("#json-output");

let previewUrls = [];
let lastResult = null;
let lastProductIdSent = "";

function selectedFiles() {
  return Array.from(imageInput.files || []);
}

function setMessage(text, type = "") {
  message.textContent = text;
  message.className = `message ${type}`.trim();
}

function clearPreviews() {
  previewUrls.forEach((url) => URL.revokeObjectURL(url));
  previewUrls = [];
  previewGrid.innerHTML = "";
}

function validateFiles(files) {
  if (files.length < 6) {
    return "Please select at least 6 images.";
  }

  if (files.length > 10) {
    return "Please select no more than 10 images.";
  }

  const invalidFile = files.find((file) => !file.type.startsWith("image/"));
  if (invalidFile) {
    return `"${invalidFile.name}" is not an image file.`;
  }

  return "";
}

function renderPreviews(files) {
  clearPreviews();
  const viewTypeByIndex = new Map((lastResult?.view_types || []).map((item) => [item.image_index, item]));
  const debugByIndex = new Map((lastResult?.image_debug || []).map((item) => [item.image_index, item]));
  const flagByIndex = new Map((lastResult?.flagged_images || []).map((item) => [item.image_index, item]));

  files.forEach((file, index) => {
    const url = URL.createObjectURL(file);
    previewUrls.push(url);
    const viewType = viewTypeByIndex.get(index);
    const debug = debugByIndex.get(index);
    const flag = flagByIndex.get(index);

    const card = document.createElement("div");
    card.className = "preview-card";

    const image = document.createElement("img");
    image.src = url;
    image.alt = `Selected product image ${index + 1}`;

    const label = document.createElement("span");
    label.textContent = file.name;

    const meta = document.createElement("div");
    meta.className = "preview-meta";
    meta.innerHTML = `
      <div>Image index: ${index}</div>
      <div>Filename: ${file.name}</div>
      <div>View type: ${viewType?.view_type || "pending"}</div>
      <div>Crop strategy: ${debug?.crop_strategy || "pending"}</div>
      <div>Flagged: ${flag ? `yes (${flag.severity})` : "no"}</div>
    `;

    card.append(image, label, meta);
    previewGrid.appendChild(card);
  });
}

function updateSelectionState() {
  const files = selectedFiles();
  imageCount.textContent = `${files.length} image${files.length === 1 ? "" : "s"} selected`;
  renderPreviews(files);

  const error = files.length ? validateFiles(files) : "";
  setMessage(error, error ? "error" : "");
}

function formatNumber(value) {
  return typeof value === "number" ? value.toFixed(4) : "n/a";
}

function renderSummary(data) {
  summaryGrid.innerHTML = "";
  const scores = data.scores || {};

  const metrics = [
    ["Status", data.status || "n/a"],
    ["Raw Score", formatNumber(scores.raw_consistency_score)],
    ["Main View Score", formatNumber(scores.main_view_score)],
    ["Detail Support", formatNumber(scores.detail_support_score)],
    ["Color Score", formatNumber(scores.color_consistency_score)],
    ["Robust Score", formatNumber(scores.robust_consistency_score)],
    ["Image Count", data.image_count ?? "n/a"],
  ];

  metrics.forEach(([label, value]) => {
    const metric = document.createElement("div");
    metric.className = "metric";
    metric.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    summaryGrid.appendChild(metric);
  });
}

function renderFlaggedImages(items = []) {
  flaggedImages.innerHTML = "";

  if (!items.length) {
    flaggedImages.textContent = "No flagged images reported.";
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "list-item";
    row.innerHTML = `
      <strong>Image ${item.image_index} (${item.view_type}, ${item.severity})</strong>
      <div>${item.reason}</div>
      <div>${item.recommended_action}</div>
    `;
    flaggedImages.appendChild(row);
  });
}

function renderDominantColors(items = []) {
  dominantColors.innerHTML = "";

  if (!items.length) {
    dominantColors.textContent = "No dominant colors returned.";
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "color-item";

    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.backgroundColor = item.color_hex;

    const text = document.createElement("span");
    text.textContent = `Image ${item.image_index}: ${item.color_hex} (${item.color_rgb?.join(", ") || "rgb n/a"})`;

    row.append(swatch, text);
    dominantColors.appendChild(row);
  });
}

function renderConditionEstimate(condition = {}) {
  conditionEstimate.innerHTML = "";

  const rows = [
    `Label: ${condition.label || "unknown"}`,
    `Confidence: ${formatNumber(condition.confidence)}`,
    condition.note || "",
  ].filter(Boolean);

  rows.forEach((text) => {
    const row = document.createElement("div");
    row.className = "list-item";
    row.textContent = text;
    conditionEstimate.appendChild(row);
  });
}

function renderMatrix(matrix = []) {
  matrixWrap.innerHTML = "";

  if (!matrix.length) {
    matrixWrap.textContent = "No matrix returned.";
    return;
  }

  const table = document.createElement("table");
  table.className = "matrix-table";

  const header = document.createElement("tr");
  header.innerHTML = "<th>Image</th>" + matrix.map((_, index) => `<th>${index}</th>`).join("");
  table.appendChild(header);

  matrix.forEach((row, index) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${index}</td>` + row.map((value) => `<td>${formatNumber(value)}</td>`).join("");
    table.appendChild(tr);
  });

  matrixWrap.appendChild(table);
}

function renderDebugCrops(items = []) {
  cropDebugGrid.innerHTML = "";

  if (!items.length) {
    cropDebugGrid.textContent = "No debug crops returned.";
    return;
  }

  items.forEach((item) => {
    const card = document.createElement("a");
    card.className = "crop-card";
    card.href = item.url;
    card.target = "_blank";
    card.rel = "noreferrer";

    const image = document.createElement("img");
    image.src = item.url;
    image.alt = `Debug crop for image ${item.image_index}`;

    const label = document.createElement("span");
    label.textContent = `Crop ${item.image_index}`;

    card.append(image, label);
    cropDebugGrid.appendChild(card);
  });
}

function renderViewTypes(items = []) {
  viewTypesWrap.innerHTML = "";

  if (!items.length) {
    viewTypesWrap.textContent = "No view type estimates returned.";
    return;
  }

  const table = document.createElement("table");
  table.className = "matrix-table";
  table.innerHTML = `
    <tr>
      <th>Image</th>
      <th>View Type</th>
      <th>Confidence</th>
      <th>Note</th>
    </tr>
  `;

  items.forEach((item) => {
    const row = document.createElement("tr");
    [item.image_index, item.view_type, formatNumber(item.confidence), item.note || ""].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    });
    table.appendChild(row);
  });

  viewTypesWrap.appendChild(table);
}

function renderImageDebug(items = []) {
  imageDebugWrap.innerHTML = "";

  if (!items.length) {
    imageDebugWrap.textContent = "No image debug info returned.";
    return;
  }

  const table = document.createElement("table");
  table.className = "matrix-table";
  table.innerHTML = `
    <tr>
      <th>Image</th>
      <th>Filename</th>
      <th>Detector Used</th>
      <th>Detection Found</th>
      <th>Confidence</th>
      <th>BBox</th>
      <th>Fallback Used</th>
      <th>Crop Strategy</th>
      <th>Rejected Reason</th>
    </tr>
  `;

  items.forEach((item) => {
    const row = document.createElement("tr");
    const fallbackText = item.fallback_used ? "Yes" : "No";
    const detectionText = item.detection_found ? "Yes" : "No";
    const detectorText = item.detector_used ? "Yes" : "No";
    const bboxText = item.bbox ? `[${item.bbox.join(", ")}]` : "None";
    [
      item.image_index,
      item.original_filename || "n/a",
      detectorText,
      detectionText,
      formatNumber(item.detection_confidence),
      bboxText,
      fallbackText,
      item.crop_strategy || "n/a",
      item.rejected_detection_reason || "",
    ].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    });

    if (item.fallback_used) {
      row.classList.add("fallback-row");
    }

    table.appendChild(row);
  });

  imageDebugWrap.appendChild(table);
}

function renderResult(data) {
  lastResult = data;
  resultPanel.classList.remove("hidden");
  statusBadge.textContent = data.status || "unknown";
  statusBadge.className = `status-badge ${data.status || ""}`.trim();
  resultNote.textContent = data.note || "";
  productIdSent.textContent = lastProductIdSent || "None";
  productIdReturned.textContent = data.product_id || "None";

  renderSummary(data);
  renderFlaggedImages(data.flagged_images);
  renderDominantColors(data.dominant_colors);
  renderConditionEstimate(data.condition_estimate);
  renderMatrix(data.pairwise_similarity_matrix);
  renderDebugCrops(data.crop_debug_urls);
  renderViewTypes(data.view_types);
  renderImageDebug(data.image_debug);
  renderPreviews(selectedFiles());
  jsonOutput.textContent = JSON.stringify(data, null, 2);
}

function parseError(payload, response) {
  if (payload && typeof payload.detail === "string") {
    return payload.detail;
  }

  if (payload && Array.isArray(payload.detail)) {
    return payload.detail.map((item) => item.msg || JSON.stringify(item)).join(" ");
  }

  return `Request failed with status ${response.status}.`;
}

async function submitValidation(event) {
  event.preventDefault();

  const files = selectedFiles();
  const validationError = validateFiles(files);
  if (validationError) {
    setMessage(validationError, "error");
    return;
  }

  const formData = new FormData();
  const productId = productIdInput.value.trim();
  lastProductIdSent = productId;
  if (productId) {
    formData.append("product_id", productId);
  }

  files.forEach((file) => {
    formData.append("images", file);
  });

  validateButton.disabled = true;
  setMessage("Processing images...", "info");

  try {
    const response = await fetch("/validate-product-images", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(parseError(payload, response));
    }

    renderResult(payload);
    setMessage("Validation completed.", "info");
  } catch (error) {
    setMessage(error.message || "Unexpected error while validating images.", "error");
  } finally {
    validateButton.disabled = false;
  }
}

imageInput.addEventListener("change", updateSelectionState);
form.addEventListener("submit", submitValidation);
