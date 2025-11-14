import os
import io
import base64
from functools import lru_cache
from typing import Tuple, List, Any
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------- CONFIG FIXA ----------
CHECKPOINT = "checkpoints/best-train.ckpt"
NUM_CLASSES = 5
TARGET_SIZE = (512, 512)
ALPHA = 0.5
ENCODER_NAME = "efficientnet-b4"
ANALYSIS_API_URL = "https://multimedia-serum-suppliers-ended.trycloudflare.com/analyze"
TIMEOUT_SECONDS = 30
# --------------------------------

app = FastAPI(title="Segmentation + Remote Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://0.0.0.0:8000",
        "*", # development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

traducao = {
    "drydown": "Seca / Ressecamento",
    "nutrient_deficiency": "Deficiência de nutrientes",
    "water": "Água",
    "planter_skip": "Falha no plantio"
}

# ------------------ modelo em cache ------------------
@lru_cache(maxsize=1)
def _load_model() -> Tuple[torch.nn.Module, torch.device]:
    """Carrega o modelo (uma vez) e retorna (model, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint não encontrado: {CHECKPOINT}")

    model = smp.UnetPlusPlus(encoder_name=ENCODER_NAME, classes=NUM_CLASSES)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model, device

def segment_from_upload_bytes(contents: bytes) -> np.ndarray:
    model, device = _load_model()

    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Não foi possível decodificar a imagem enviada.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    preprocess_input = smp.encoders.get_preprocessing_fn(ENCODER_NAME, pretrained="imagenet")
    img_resized = cv2.resize(img_rgb, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    img_pre = preprocess_input(img_resized)
    tensor = torch.from_numpy(img_pre.transpose(2, 0, 1)).unsqueeze(0).to(device).float()

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).cpu().numpy()[0].astype(np.uint8)

    pred_original = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # cores fixas (RGB)
    BASE_COLORS = [
        (0, 0, 0),
        (220, 20, 60),
        (34, 139, 34),
        (30, 144, 255),
        (255, 165, 0),
        (148, 0, 211),
    ]
    while len(BASE_COLORS) <= NUM_CLASSES:
        np.random.seed(len(BASE_COLORS) * 123)
        BASE_COLORS.append(tuple((np.random.randint(0, 255, 3)).tolist()))

    overlay = img_rgb.copy().astype(np.float32)
    for cls_id in range(1, NUM_CLASSES):
        mask = (pred_original == cls_id).astype(np.uint8) * 255
        if mask.sum() == 0:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = BASE_COLORS[cls_id][:3]
        colored_mask = np.zeros_like(overlay, dtype=np.uint8)
        colored_mask[mask == 255] = color
        overlay[mask == 255] = (1 - ALPHA) * overlay[mask == 255] + ALPHA * colored_mask[mask == 255]
        cv2.drawContours(overlay, contours, -1, color, thickness=2, lineType=cv2.LINE_AA)

    output_img = np.clip(overlay, 0, 255).astype(np.uint8)
    return output_img

# ------------------ util: encode image para data URL ------------------
def image_to_data_url_rgb(img_rgb: np.ndarray, ext: str = ".png") -> str:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(ext, bgr)
    if not success:
        raise RuntimeError("Falha ao encoder a imagem.")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    mime = "image/png" if ext.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

# ------------------ Endpoint que integra tudo ------------------
@app.post("/analyze")
async def analyze_endpoint(image: UploadFile = File(...)):
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    try:
        processed_img_rgb = segment_from_upload_bytes(contents)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro durante segmentação: {e}")

    remote_response_json: dict = {}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            filename = getattr(image, "filename", "upload.png") or "upload.png"
            content_type = image.content_type if hasattr(image, "content_type") else "image/png"
            files = {"file": (filename, contents, content_type)}
            resp = await client.post(ANALYSIS_API_URL, files=files)
            resp.raise_for_status()
            remote_response_json = resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Erro do serviço remoto: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao contatar serviço remoto: {e}")

    label = remote_response_json.get("labels") or remote_response_json.get("label") or []
    mensagem = remote_response_json.get("message")

    try:
        image_data_url = image_to_data_url_rgb(processed_img_rgb, ext=".png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao codificar imagem: {e}")

    
    return JSONResponse(
        content={
            "label": [traducao.get(l) for l in label],
            "mensagem": mensagem,
            "imagem": image_data_url,
            "raw_remote_response": remote_response_json,  # opcional: inclui resposta completa remota para debug
        }
    )
