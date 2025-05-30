# src/preprocessing/chunker.py
import json, os
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
IN_FILE = "C:/Users/zaibu/Desktop/FAST/8th_semester/GEN AI/multimodal_rag/Data/chunks.jsonl"
OUT_FILE = "C:/Users/zaibu/Desktop/FAST/8th_semester/GEN AI/multimodal_rag/Data/chunks_chunked.jsonl"
MAX_TOKENS = 500  # rough

def chunk_text(text):
    words = text.split()
    for i in range(0, len(words), MAX_TOKENS):
        yield " ".join(words[i:i+MAX_TOKENS])

with open(IN_FILE) as fin, open(OUT_FILE, "w") as fout:
    for line in fin:
        obj = json.loads(line)
        if obj["type"] == "text":
            for i, sub in enumerate(chunk_text(obj["content"])):
                obj2 = obj.copy()
                obj2["chunk_id"] = f"{obj['doc']}-{obj['page']}-txt-{i}"
                obj2["content"] = sub
                fout.write(json.dumps(obj2)+"\n")
        else:
            # keep images and OCR as single chunks
            obj["chunk_id"] = f"{obj['doc']}-{obj['page']}-img"
            fout.write(json.dumps(obj)+"\n")
