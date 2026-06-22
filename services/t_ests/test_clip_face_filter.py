# test_clip_face_filter.py

from pathlib import Path
from collections import Counter, defaultdict

import torch
from PIL import Image
from tqdm import tqdm
import open_clip

from db.session import SessionLocal
from db.models import DBFace, DBFreeze


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

CROP_MARGIN = 0.20
MIN_CONFIDENCE = 0.50

CATEGORIES = {
    "real_human": [
        "a real human face",
        "a natural human face in video footage",
        "a real person in a video frame",
        "a face on a television screen",
    ],
    "cartoon": [
        "a cartoon face",
        "an animated character face",
        "a drawn illustration of a face",
        "a mannequin face",
        "a plastic doll face",
        "an artificial shop mannequin head",
        "a statue face",
        "a wax figure face",
        "a sculpture of a human face",
        "a printed poster with a face",
    ],
    "ai_face": [
        "a 3d rendered artificial face",
        "an ai generated face",
    ],
}

SUSPICIOUS_CATEGORIES = {
    "cartoon",
    "screen_poster_render",
}


def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model = model.to(DEVICE)
    model.eval()

    return model, preprocess, tokenizer


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
    if bbox is None:
        raise ValueError("bbox is None")

    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values, got: {bbox}")

    x1, y1, x2, y2 = bbox

    return int(x1), int(y1), int(x2), int(y2)


def crop_face(image: Image.Image, bbox):
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
    best_score = category_scores[best_category]

    return best_category, best_score, dict(category_scores)


def get_freeze_image_path(freeze: DBFreeze) -> Path:
    return Path(freeze.freeze_path)


def main():
    print(f"DEVICE: {DEVICE}")
    print("Loading CLIP...")

    model, preprocess, tokenizer = load_clip()
    text_features, prompt_categories = prepare_text_features(model, tokenizer)

    db = SessionLocal()

    stats = Counter()
    suspicious = []
    errors = []

    try:
        faces = (
            db.query(DBFace)
            .filter(DBFace.confidence.is_(None))
            .all()
        )

        print(f"Faces with confidence IS NULL: {len(faces)}")

        freeze_cache = {}

        for face in tqdm(faces):
            try:
                freeze_id = face.freeze_id

                if freeze_id not in freeze_cache:
                    freeze_cache[freeze_id] = (
                        db.query(DBFreeze)
                        .filter(DBFreeze.id == freeze_id)
                        .first()
                    )

                freeze = freeze_cache[freeze_id]

                if freeze is None:
                    raise ValueError(f"Freeze not found: freeze_id={freeze_id}")

                image_path = get_freeze_image_path(freeze)

                if not image_path.exists():
                    raise FileNotFoundError(f"Freeze image not found: {image_path}")

                image = Image.open(image_path).convert("RGB")
                crop = crop_face(image, face.bbox)

                category, score, all_scores = classify_crop(
                    crop=crop,
                    model=model,
                    preprocess=preprocess,
                    text_features=text_features,
                    prompt_categories=prompt_categories,
                )

                stats[category] += 1

                low_confidence = score < MIN_CONFIDENCE
                suspicious_category = category in SUSPICIOUS_CATEGORIES

                if suspicious_category or low_confidence:
#                if category == "mannequin_doll":
                    suspicious.append({
                        "face_id": face.id,
                        "embedding_id": face.embedding_id,
                        "freeze_id": face.freeze_id,
                        "person_name": face.person.name if face.person else None,
                        "category": category,
                        "score": round(score, 4),
                        "reason": "low_clip_confidence" if low_confidence else category,
                    })

            except Exception as e:
                stats["error"] += 1
                errors.append({
                    "face_id": getattr(face, "id", None),
                    "embedding_id": getattr(face, "embedding_id", None),
                    "freeze_id": getattr(face, "freeze_id", None),
                    "error": str(e),
                })

    finally:
        db.close()

    print("\n==============================")
    print("СТАТИСТИКА ПО КАТЕГОРІЯХ")
    print("==============================")

    for category in CATEGORIES:
        print(f"{category}: {stats[category]}")

    print(f"errors: {stats['error']}")

    print("\n==============================")
    print("ПІДОЗРІЛІ ОБЛИЧЧЯ")
    print("==============================")

    for item in suspicious:
        print(
            f"embedding_id={item['embedding_id']} | "
            f"face_id={item['face_id']} | "
            f"freeze_id={item['freeze_id']} | "
            f"person_name={item['person_name']} | "
            f"suspicion={item['reason']} | "
            f"category={item['category']} | "
            f"score={item['score']}"
        )

    print("\n==============================")
    print("ПОМИЛКИ")
    print("==============================")

    for item in errors[:100]:
        print(
            f"face_id={item['face_id']} | "
            f"embedding_id={item['embedding_id']} | "
            f"freeze_id={item['freeze_id']} | "
            f"error={item['error']}"
        )

    print("\n==============================")
    print("ПІДСУМОК")
    print("==============================")
    print(f"Total suspicious: {len(suspicious)}")
    print(f"Total errors: {len(errors)}")


if __name__ == "__main__":
    main()