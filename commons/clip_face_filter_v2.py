from collections import defaultdict

import torch
from PIL import Image

from commons.clip_face_categories import DEFAULT_FACE_CATEGORY


CROP_MARGIN = 0.50
CONFIDENT_CATEGORY_THRESHOLD = 0.75
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid cropped bbox: {(x1, y1, x2, y2)}")

    return image.crop((x1, y1, x2, y2)).convert("RGB")


def classify_crop(crop, model, preprocess, text_features, prompt_categories):
    image_tensor = preprocess(crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = probs[0].detach().cpu().tolist()

    category_scores = defaultdict(float)

    for prob, category_name in zip(probs, prompt_categories):
        category_scores[category_name] += float(prob)

    category_scores = {
        category_name: round(float(score), 4)
        for category_name, score in category_scores.items()
    }

    best_clip_category = max(category_scores, key=category_scores.get)
    best_clip_score = category_scores[best_clip_category]

    if best_clip_score < CONFIDENT_CATEGORY_THRESHOLD:
        final_category = DEFAULT_FACE_CATEGORY
        category_score = best_clip_score
    else:
        final_category = best_clip_category
        category_score = best_clip_score

    return {
        "category": final_category,
        "category_score": round(float(category_score), 4),
        "best_clip_category": best_clip_category,
        "best_clip_score": round(float(best_clip_score), 4),
        "clip_scores": category_scores,
    }


def analyze_face_category(
    image: Image.Image,
    bbox,
    model,
    preprocess,
    text_features,
    prompt_categories,
):
    crop = crop_face_for_clip(image, bbox)

    return classify_crop(
        crop=crop,
        model=model,
        preprocess=preprocess,
        text_features=text_features,
        prompt_categories=prompt_categories,
    )
