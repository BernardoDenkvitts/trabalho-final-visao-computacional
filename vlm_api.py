from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch
import io
import json
import re
import uvicorn
from typing import List


CLASSES = ["drydown", "nutrient_deficiency", "planter_skip", "water"]

PT_TO_EN = {
    "secamento": "drydown",
    "senescência": "drydown",
    "deficiência nutricional": "nutrient_deficiency",
    "deficiencia nutricional": "nutrient_deficiency",
    "falha de plantio": "planter_skip",
    "falha na plantação": "planter_skip",
    "acúmulo de água": "water",
    "acumulo de água": "water"
}

SYSTEM_PROMPT = """Você é um especialista em agronomia e análise de imagens aéreas agrícolas. Analise a imagem identificando problemas agrícolas com base em padrões visuais característicos.

Classes válidas (use apenas estas):
• drydown: zonas amareladas ou marrons indicando secamento natural (senescência ou maturação)
• nutrient_deficiency: áreas claras ou esverdeadas com coloração irregular, sem falhas de plantio (carência de nutrientes)
• planter_skip: linhas vazias ou falhas retilíneas entre fileiras (problema de plantio)
• water: Manchas extensas de baixa reflectância, campo alagado ou com acúmulo de água (inundação)

Análise Espacial: Observe distribuição, padrão e relação com as fileiras de plantio.

Formato de resposta (JSON puro, sem markdown):
{"labels": ["classe1"], "message": "descrição em português"}

Se não tem problemas: {"labels": [], "message": "Nenhum problema identificado."}"""

TEMPLATES_SINGLE = {
    "drydown": "Observo faixas de tonalidade marrom-clara distribuídas de forma irregular ou Áreas secas seguindo padrão de distribuição não-linear. Característico de senescência natural da lavoura.",
    "nutrient_deficiency": "Identifico variações sutis no vigor e na tonalidade da cultura ou Regiões com aspecto esverdeado e pálido, distribuídas de forma irregular. Indica possível deficiência de nutrientes.",
    "planter_skip": "Há Intervalos vazios mantendo equidistância entre plantas adjacentes ou Linhas vazias com espaçamento regular. Padrão retilíneo típico de problema de semeadura.",
    "water": "Detecto manchas escuras contínuas indicando acúmulo de água no campo."
}

TEMPLATES_PAIR = {
    ("nutrient_deficiency", "planter_skip"): "Verifico deficiência nutricional nas áreas plantadas e também falhas de semeadura em algumas linhas."
}

class VLMResponse(BaseModel):
    labels: List[str]
    message: str

app = FastAPI(
    title="Agricultural VLM Analysis API",
    description="API para análise de imagens agrícolas usando Qwen3-VL",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= VARIÁVEIS GLOBAIS =============
model = None
processor = None
tokenizer = None
few_shot_messages = []
FEW_SHOT_DIR = "few_shot_images"
IMAGENS_FEW_SHOT = {
    "drydown": "A2YGL9XDW_6561-5017-7073-5529.jpg", 
    "nutrient_deficiency": "XBL7ZHEMA_5840-745-6352-1257.jpg",
    "planter_skip": "8PNFJERGL_3889-443-4401-955.jpg",
    "water": "J3GVGXF1W_3380-12636-3892-13148.jpg",
    "nutrient_deficiency|planter_skip": "DJU48KGYH_3766-12675-4278-13187.jpg"
}


def create_realistic_message(labels):
    if len(labels) == 1:
        return TEMPLATES_SINGLE.get(labels[0])
    else:
        par = tuple(sorted(labels))
        if par in TEMPLATES_PAIR:
            return TEMPLATES_PAIR.get(par)
        else:
            return f"Problemas detectados: {', '.join(labels)}."


def extract_json(text):
    text = text.strip()
    
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    json_match = re.search(r'\{.*?"labels".*?\[.*?\].*?"message".*?\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)
    
    try:
        parsed = json.loads(text)
        if "labels" in parsed and "message" in parsed:
            valid_labels = [lbl for lbl in parsed["labels"] if lbl in CLASSES]
            parsed["labels"] = sorted(set(valid_labels))
            return parsed
    except:
        pass
    
    return fallback_label_search(text)


def fallback_label_search(text):
    """Fallback mais inteligente"""
    text_lower = text.lower()
    detected = []
    
    for cls in CLASSES:
        if cls in text_lower or cls.replace("_", " ") in text_lower:
            detected.append(cls)

    for pt_term, en_term in PT_TO_EN.items():
        if pt_term in text_lower and en_term not in detected:
            detected.append(en_term)
    
    return {
        "labels": sorted(set(detected)),
        "message": text if text else "Erro no parsing"
    }
    

def build_inputs_images_only(messages):
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    imgs = []
    for turn in messages:
        if isinstance(turn.get("content"), list):
            for item in turn["content"]:
                if item.get("type") == "image":
                    imgs.append(item["image"])
    
    enc = processor(text=[chat_text], images=imgs, padding=True, return_tensors="pt")
    return enc


def move_batch_to_model(enc, model):
    text_device = model.get_input_embeddings().weight.device
    target_dtype = model.dtype

    out = {}
    for k, v in enc.items():
        if torch.is_tensor(v):
            if k == "pixel_values":
                out[k] = v.to(device=text_device, dtype=target_dtype, non_blocking=True)
            else:
                out[k] = v.to(device=text_device, non_blocking=True)
        else:
            out[k] = v
    return out


def generate_with_retry(messages, max_retries=2):
    enc = build_inputs_images_only(messages)
    enc = move_batch_to_model(enc, model)

    parsed = {"labels": [], "message": ""}
    out_text = ""

    do_sample = True
    temperature = 0.45
    top_p = 0.85  # Nucleus sampling mais restritivo
    top_k = 50  # Limita às 50 opções mais prováveis

    print("Executando inferencia...")
    with torch.inference_mode():
        gen = model.generate(
            **enc,
            max_new_tokens=300,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,  # Mais suave para não forçar palavras estranhas
            no_repeat_ngram_size=3,
        )

        gen_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(enc["input_ids"], gen)]
        out_text = processor.batch_decode(
            gen_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        parsed = extract_json(out_text)
        msg_l = parsed.get("message", "").lower()

        if parsed.get("labels") or "nenhum" in msg_l or "saudável" in msg_l:
            return parsed, out_text

    return parsed, out_text


def build_messages(target_img):
    messages = [
        {"role": "user", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        *few_shot_messages,
        {
            "role": "user",
            "content": [
                {"type": "image", "image": target_img},
                {"type": "text", "text": "Analise esta imagem:"}
            ],
        },
    ]
    return messages


def shot_to_messages(img, classes_str):
    """Converte uma imagem few-shot em par de mensagens user/assistant"""
    labels = classes_str.split("|")
    message = create_realistic_message(labels) or f"Problemas detectados: {', '.join(sorted(labels))}."
    assistant_json = json.dumps({"labels": labels, "message": message}, ensure_ascii=False)

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Analise esta imagem:"}
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_json}
            ],
        },
    ]


def load_few_shot_examples():
    """Carrega as imagens few-shot e monta os exemplos"""
    from pathlib import Path
    
    few_messages = []
    few_dir = Path(FEW_SHOT_DIR)
    
    if not few_dir.exists():
        print(f"Diretório {FEW_SHOT_DIR} não encontrado. Few-shot desabilitado.")
        return []
    
    for classes_str, img_name in IMAGENS_FEW_SHOT.items():
        img_path = few_dir / img_name
        
        if not img_path.exists():
            print(f"Imagem {img_name} não encontrada, pulando...")
            continue
        
        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            few_messages.extend(shot_to_messages(img, classes_str))
            print(f"Carregado exemplo: {classes_str} ({img_name})")
        except Exception as e:
            print(f"Erro ao carregar {img_name}: {e}")
    
    return few_messages


@app.on_event("startup")
async def load_model():
    """Carrega o modelo na inicialização da API"""
    global model, processor, tokenizer, few_shot_messages
    
    print("Carregando modelo Qwen3-VL-32B-Instruct...")
    
    MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        attn_implementation="flash_attention_2",
        device_map="auto",
        quantization_config=bnb_config
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = processor.tokenizer
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    few_shot_messages = load_few_shot_examples()

    print("Modelo carregado!")


@app.get("/")
async def root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "Agricultural VLM Analysis API",
        "status": "online",
        "model": "Qwen3-VL-32B-Instruct",
        "endpoints": {
            "analyze": "/analyze (POST) - Analisa imagem agrícola",
            "health": "/health (GET) - Verifica status da API"
        }
    }


@app.get("/health")
async def health_check():
    """Verifica o status da API e do modelo"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "processor_loaded": processor is not None
    }


@app.post("/analyze", response_model=VLMResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analisa uma imagem agrícola usando o VLM Qwen3
    
    Args:
        file: Arquivo de imagem (PNG, JPG, JPEG)
    
    Returns:
        VLMResponse: JSON com labels (lista de problemas) e message (descrição)
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Modelo não está carregado")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Arquivo deve ser uma imagem (PNG, JPG, JPEG)"
        )
    
    try:
        # Ler e processar a imagem
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Converter para RGB se necessário
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Construir mensagens com a imagem
        messages = build_messages(image)
        
        # Gerar análise com retry
        parsed, raw_output = generate_with_retry(messages, max_retries=2)
        
        return VLMResponse(
            labels=parsed.get("labels", []),
            message=parsed.get("message", "Erro ao processar imagem")
        )
    
    except Exception as e:
        print(f"❌ Erro ao processar: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar imagem: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )