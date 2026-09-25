import logging

import open_clip
import torch
from insightface.app import FaceAnalysis

from commons.clip_face_categories import CLIP_FACE_CATEGORIES


FACE_DET_SIZE = 640

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


_INSIGHTFACE_CACHE = None
_CLIP_CACHE = None


def load_insightface():
    logging.info("Loading InsightFace buffalo_l...")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(FACE_DET_SIZE, FACE_DET_SIZE))
    logging.info("InsightFace buffalo_l loaded.")
    return app


def get_insightface():
    global _INSIGHTFACE_CACHE
    if _INSIGHTFACE_CACHE is None:
        _INSIGHTFACE_CACHE = load_insightface()
    return _INSIGHTFACE_CACHE


def load_clip():
    logging.info("Loading CLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()

    text_features, prompt_categories = prepare_text_features(model, tokenizer)
    logging.info("CLIP loaded.")
    return model, preprocess, text_features, prompt_categories


def get_clip():
    global _CLIP_CACHE

    if _CLIP_CACHE is None:
        _CLIP_CACHE = load_clip()

    return _CLIP_CACHE


def prepare_text_features(model, tokenizer):
    prompts = []
    prompt_categories = []

    for category_name, category_prompts in CLIP_FACE_CATEGORIES.items():
        for prompt in category_prompts:
            prompts.append(prompt)
            prompt_categories.append(category_name)

    tokens = tokenizer(prompts).to(DEVICE)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features, prompt_categories
