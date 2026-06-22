from collections import defaultdict
import torch
import open_clip
from PIL import Image


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

CROP_MARGIN = 0.20
MIN_CLIP_CONFIDENCE = 0.55
_CLIP_CACHE = None


CATEGORIES = {
    "real_human": [
        "a real human face",
        "a natural human face in video footage",
        "a real person in a video frame",
        "a human face from news footage",
    ],
    "screen_face": [
        "a face on a television screen",
        "a face displayed on a monitor",
        "a projected human face",
    ],
    "printed_face": [
        "a printed poster with a face",
        "a printed photograph of a person",
        "a magazine photo of a face",
    ],
    "cartoon": [
        "a cartoon face",
        "an animated character face",
        "a drawn illustration of a face",
    ],
    "mannequin": [
        "a mannequin face",
        "a plastic doll face",
        "an artificial shop mannequin head",
    ],
    "statue": [
        "a statue face",
        "a wax figure face",
        "a sculpture of a human face",
    ],
    "ai_face": [
        "a 3d rendered artificial face",
        "an ai generated face",
        "a synthetic human face",
    ],
}


SUSPICIOUS_CATEGORIES = {
    "screen_face",
    "printed_face",
    "cartoon",
    "mannequin",
    "statue",
    "ai_face",
}


def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model = model.to(DEVICE)
    model.eval()

    text_features, prompt_categories = prepare_text_features(model, tokenizer)

    return model, preprocess, text_features, prompt_categories


def get_clip():
    global _CLIP_CACHE

    if _CLIP_CACHE is None:
        _CLIP_CACHE = load_clip()

    return _CLIP_CACHE


def prepare_text_features(model, tokenizer):
    prompts = []
    prompt_categories = []

    for category, category_prompts in CATEGORIES.items():
        for prompt in category_prompts:
            prompts.append(prompt)
            prompt_categories.append(category)

    tokens = tokenizer(prompts).to(DEVICE)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features, prompt_categories


def parse_bbox(bbox):
    if bbox is None or len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox}")

    x1, y1, x2, y2 = bbox
    return int(x1), int(y1), int(x2), int(y2)


def crop_face_for_clip(image: Image.Image, bbox):
    x1, y1, x2, y2 = parse_bbox(bbox)

    image_w, image_h = image.size

    face_w = x2 - x1
    face_h = y2 - y1

    if face_w <= 0 or face_h <= 0:
        raise ValueError(f"Invalid bbox size: {bbox}")

    margin_x = int(face_w * CROP_MARGIN)
    margin_y = int(face_h * CROP_MARGIN)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image_w, x2 + margin_x)
    y2 = min(image_h, y2 + margin_y)

    return image.crop((x1, y1, x2, y2)).convert("RGB")


def classify_crop(crop, model, preprocess, text_features, prompt_categories):
    image_tensor = preprocess(crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = probs[0].detach().cpu().tolist()

    category_scores = defaultdict(float)

    for prob, category in zip(probs, prompt_categories):
        category_scores[category] += prob

    best_category = max(category_scores, key=category_scores.get)
    best_score = float(category_scores[best_category])

    return {
        "category": best_category,
        "score": round(best_score, 4),
        "all_scores": {
            key: round(value, 4)
            for key, value in category_scores.items()
        },
    }


def check_face_suspicion(
    image: Image.Image,
    bbox,
    model,
    preprocess,
    text_features,
    prompt_categories,
):
    crop = crop_face_for_clip(image, bbox)

    result = classify_crop(
        crop=crop,
        model=model,
        preprocess=preprocess,
        text_features=text_features,
        prompt_categories=prompt_categories,
    )

    category = result["category"]
    score = result["score"]

    if category in SUSPICIOUS_CATEGORIES:
        result["is_suspicious"] = True
        result["reason"] = category
        return result

    if category == "real_human" and score < MIN_CLIP_CONFIDENCE:
        result["is_suspicious"] = True
        result["reason"] = "low_clip_confidence"
        return result

    result["is_suspicious"] = False
    result["reason"] = None

    return result
