# src/search_faiss.py
import json, faiss, numpy as np, base64, io, torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# ─── LOAD INDEX + METADATA ────────────────────────────────────────────
index = faiss.read_index("faiss_index/index.faiss")
d      = index.d  # the dimension we padded to when building
with open("faiss_index/metadata.json", "r", encoding="utf8") as f:
    metadata = json.load(f)

# ─── MODELS ─────────────────────────────────────────────────────────────
txt_model = SentenceTransformer("all-MiniLM-L6-v2")

# Optional: image support if you installed CLIP
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dim_img    = clip_model.config.projection_dim
except:
    clip_model = None
    dim_img    = None

def search_text(query, k=5):
    # 1) get 384-dim text embedding
    emb = txt_model.encode(query, normalize_embeddings=True)
    # 2) pad to d for FAISS
    if emb.shape[0] < d:
        emb = np.pad(emb, (0, d - emb.shape[0]), mode="constant")
    emb = emb.astype("float32")
    # 3) search
    D, I = index.search(np.array([emb]), k)
    # 4) build results
    return [
        {
            "chunk_id": metadata[i]["chunk_id"],
            "type":     metadata[i]["type"],
            "caption":  metadata[i]["content"],
            "score":    float(D[0, rank])
        }
        for rank, i in enumerate(I[0])
    ]

def search_image_b64(b64str, k=5):
    if clip_model is None:
        raise RuntimeError("CLIP model not loaded")
    img = Image.open(io.BytesIO(base64.b64decode(b64str))).convert("RGB")
    inputs = clip_proc(images=img, return_tensors="pt").to(clip_model.device)
    with torch.no_grad():
        feat = clip_model.get_image_features(**inputs)[0].cpu().numpy()
    # normalize
    feat = feat / np.linalg.norm(feat)
    # pad if needed
    if feat.shape[0] < d:
        feat = np.pad(feat, (0, d - feat.shape[0]), mode="constant")
    feat = feat.astype("float32")
    D, I = index.search(np.array([feat]), k)
    return [
        {
            "chunk_id": metadata[i]["chunk_id"],
            "type":     metadata[i]["type"],
            "caption":  metadata[i]["content"],
            "score":    float(D[0, rank])
        }
        for rank, i in enumerate(I[0])
    ]
