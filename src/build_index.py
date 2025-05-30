# src/build_index.py

import os, shutil, json, io, base64
import numpy as np
import faiss, pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# ─── CONFIG ─────────────────────────────────────────────────────────
DATA_PATH        = os.path.join("Data", "chunks_chunked.jsonl")
FAISS_DIR        = "faiss_index"
INDEX_PATH       = os.path.join(FAISS_DIR, "index.faiss")
META_PATH        = os.path.join(FAISS_DIR, "metadata.json")

# ─── CLEAN OLD ──────────────────────────────────────────────────────
if os.path.exists(FAISS_DIR):
    shutil.rmtree(FAISS_DIR)
os.makedirs(FAISS_DIR, exist_ok=True)

# ─── MODELS ─────────────────────────────────────────────────────────
txt_model  = SentenceTransformer("all-MiniLM-L6-v2")
dim_txt    = txt_model.get_sentence_embedding_dimension()  # probably 384

# Optional: enable this if you want image embeddings too
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dim_img    = clip_model.config.projection_dim           # typically 512
except:
    clip_model = None
    dim_img    = dim_txt

# use the larger of the two dims
DIM = max(dim_txt, dim_img)
index = faiss.IndexFlatIP(DIM)

metadata = []

def is_base64(s):
    try:
        base64.b64decode(s)
        return True
    except:
        return False

# ─── BUILD ──────────────────────────────────────────────────────────
with open(DATA_PATH, "r", encoding="utf8") as f:
    for line in f:
        obj = json.loads(line)
        cid   = obj.get("chunk_id") or obj.get("id")
        ctype = obj.get("type", "text")
        cont  = obj.get("content", "")

        try:
            if ctype == "text":
                # embed text → 384 → pad to DIM
                emb = txt_model.encode(cont, normalize_embeddings=True)
                if DIM > dim_txt:
                    emb = np.pad(emb, (0, DIM - dim_txt))
            elif ctype == "image" and clip_model:
                if not isinstance(cont, str) or not is_base64(cont):
                    raise ValueError("Invalid base64")
                img = Image.open(io.BytesIO(base64.b64decode(cont))).convert("RGB")
                inp = clip_proc(images=img, return_tensors="pt")
                with torch.no_grad():
                    feats = clip_model.get_image_features(**{k:v.to(clip_model.device) for k,v in inp.items()})
                emb = feats[0].cpu().numpy()
                emb = emb / np.linalg.norm(emb)
                if DIM > dim_img:
                    emb = np.pad(emb, (0, DIM - dim_img))
            else:
                # skip unknown types or if CLIP not installed
                print(f"⚠️  Skipping {ctype} #{cid}")
                continue

            index.add(np.array([emb], dtype="float32"))
            metadata.append({
                "chunk_id": cid,
                "type":     ctype,
                "doc":      obj.get("doc"),
                "page":     obj.get("page"),
                "content":  cont[:200]  # store the first 200 chars
            })
            print(f"✔️  Indexed {ctype} #{cid}")

        except Exception as e:
            print(f"❌  Skipped {ctype} #{cid}: {e}")

# ─── SAVE ───────────────────────────────────────────────────────────
faiss.write_index(index, INDEX_PATH)
pd.DataFrame(metadata).to_json(META_PATH, orient="records", indent=2)
print(f"\n✅ Persisted FAISS index ({index.ntotal} vectors) + metadata.")

