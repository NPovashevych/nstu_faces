from pathlib import Path

import cv2
import numpy as np

from commons.common_model import get_insightface
from commons.commons_base import cosine_similarity, normalize_vector


MIN_W = 15
MIN_H = 15
BLUR_THRESHOLD = 15

DUPLICATE_SIMILARITY = 0.98
MAX_DIST_FROM_MEAN = 0.50
MAX_PAIRWISE_DIST = 0.72


# Читання зображення через OpenCV
def read_image(image_path: Path):
    try:
        return cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


# Розрахунок різкості зображення
def get_blur_value(image) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# Вибір найбільшого обличчя
def get_largest_face(faces):
    return max(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))


# Перевірка одного фото
def check_candidate_photo(model, image_path: Path):
    result = {
        "file_name": image_path.name,
        "readable": False,
        "width": None,
        "height": None,
        "blur": None,
        "face_count": 0,
        "embedding": None,
        "warnings": [],
    }

    image = read_image(image_path)

    if image is None:
        result["warnings"].append("image_read_error")
        return result

    result["readable"] = True

    height, width = image.shape[:2]

    result["width"] = width
    result["height"] = height

    if width < MIN_W or height < MIN_H:
        result["warnings"].append("image_too_small")
        return result

    blur = get_blur_value(image)
    result["blur"] = round(blur, 2)

    if blur < BLUR_THRESHOLD:
        result["warnings"].append("image_blurred")

    faces = model.get(image)
    result["face_count"] = len(faces)

    if not faces:
        result["warnings"].append("face_not_found")
        return result

    if len(faces) > 1:
        result["warnings"].append("multiple_faces")

    face = get_largest_face(faces)
    result["embedding"] = normalize_vector(face.embedding)

    return result


# Пошук схожих фотографій у наборі
def find_duplicate_photos(valid_photos):
    duplicates = []

    for i in range(len(valid_photos)):
        for j in range(i + 1, len(valid_photos)):
            first = valid_photos[i]
            second = valid_photos[j]

            similarity = cosine_similarity(
                first["embedding"],
                second["embedding"],
            )

            if similarity > DUPLICATE_SIMILARITY:
                duplicates.append({
                    "file_1": first["file_name"],
                    "file_2": second["file_name"],
                    "similarity": round(float(similarity), 4),
                })

    return duplicates


# Перевірка, чи всі фото належать одній персоні
def check_same_person(valid_photos):
    if len(valid_photos) < 2:
        return []

    embeddings = [photo["embedding"] for photo in valid_photos]

    mean_embedding = normalize_vector(np.mean(embeddings, axis=0))

    warnings = []

    for photo in valid_photos:
        similarity = cosine_similarity(photo["embedding"], mean_embedding)
        distance = 1 - similarity
        if distance > MAX_DIST_FROM_MEAN:
            warnings.append({
                "type": "far_from_mean",
                "file_name": photo["file_name"],
                "distance": round(float(distance), 4),
            })

    for i in range(len(valid_photos)):
        for j in range(i + 1, len(valid_photos)):
            first = valid_photos[i]
            second = valid_photos[j]

            similarity = cosine_similarity(first["embedding"], second["embedding"])
            distance = 1 - similarity
            if distance > MAX_PAIRWISE_DIST:
                warnings.append({
                    "type": "pair_distance",
                    "file_1": first["file_name"],
                    "file_2": second["file_name"],
                    "distance": round(float(distance), 4),
                })

    return warnings


# Перевірка всіх фото кандидата
def check_candidate_folder(candidate_folder: Path):
    model = get_insightface()

    photo_results = []

    for image_path in sorted(candidate_folder.iterdir()):
        if not image_path.is_file():
            continue

        result = check_candidate_photo(model=model, image_path=image_path)

        photo_results.append(result)

    valid_photos = [photo for photo in photo_results if photo["embedding"] is not None]

    duplicates = find_duplicate_photos(valid_photos)
    same_person_warnings = check_same_person(valid_photos)

    # Embedding потрібен тільки всередині сервісу
    for photo in photo_results:
        photo.pop("embedding", None)

    return {
        "photos": photo_results,
        "duplicates": duplicates,
        "same_person_warnings": same_person_warnings,
    }
