# src/qa.py

import numpy as np
from transformers import pipeline, set_seed
from search_faiss import search_text, search_image_b64

# ─── Model Setup ───────────────────────────────────────────────────────

# Use DistilGPT2 for fast local inference
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    device=-1                # -1 = CPU; change to 0 for GPU if available
)
set_seed(42)

# ─── Helpers ──────────────────────────────────────────────────────────

def format_hits(hits):
    """
    Convert a list of hit dicts into a bullet-list string of captions.
    Each hit is expected to have a 'caption' key.
    """
    lines = []
    for h in hits:
        caption = h.get("caption", "<no caption>")
        score   = h.get("score", 0.0)
        lines.append(f"• {caption}  (score: {score:.3f})")
    return "\n".join(lines)

# ─── QA Functions ─────────────────────────────────────────────────────

def answer_from_text(query: str, k: int = 5) -> dict:
    """
    Retrieve top-k text chunks for `query`, then generate an answer
    using DistilGPT2 with a few-shot + CoT prompt.
    Returns a dict with:
      - 'answer': the generated answer string
      - 'hits': the list of hit dicts
    """
    hits = search_text(query, k)
    facts = format_hits(hits)

    # Few-shot example + CoT instruction
    prompt = f"""
Example:
Facts:
• FAST University is a private research university in Islamabad, Pakistan.  (score: 0.99)
• It was founded in 2000.                                           (score: 0.95)
Question: When was FAST University founded?
Answer: FAST University was founded in 2000.

Now you:
Facts:
{facts}

Question: {query}
Answer (use only the facts above, think step-by-step):"""

    output = generator(
        prompt,
        max_new_tokens=100,   # generate up to 100 new tokens
        truncation=True,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )[0]["generated_text"]

    # Extract only the generated portion after our prompt
    answer = output[len(prompt):].strip()
    return {"answer": answer, "hits": hits}


def answer_from_image_b64(b64str: str, k: int = 5) -> dict:
    """
    Retrieve top-k image chunks for the base64 image, then generate
    a description using DistilGPT2 with a CoT prompt.
    Returns a dict with:
      - 'answer': the generated description
      - 'hits': the list of hit dicts
    """
    hits = search_image_b64(b64str, k)
    facts = format_hits(hits)

    prompt = f"""
Facts (captions):
{facts}

Describe the image in detail, using only these captions, think step-by-step:"""

    output = generator(
        prompt,
        max_new_tokens=100,
        truncation=True,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )[0]["generated_text"]

    # Extract generated portion after prompt
    desc = output[len(prompt):].strip()
    return {"answer": desc, "hits": hits}
