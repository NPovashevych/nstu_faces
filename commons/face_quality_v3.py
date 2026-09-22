# services/create_faces/face_quality_v3.py

import math

import cv2


GOOD_FACE_SIZE = 80
GOOD_BLUR = 60

MIN_ASPECT = 0.55
MAX_ASPECT = 1.55
IDEAL_ASPECT_MIN = 0.75
IDEAL_ASPECT_MAX = 1.25

MIN_EYE_DIST = 4
MIN_MOUTH_DIST = 2


def crop_face(img, bbox):
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = img.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def visible_bbox_ratio(img, bbox):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)

    full_area = max(0, x2 - x1) * max(0, y2 - y1)
    if full_area <= 0:
        return 0.0

    vx1 = max(0, x1)
    vy1 = max(0, y1)
    vx2 = min(w, x2)
    vy2 = min(h, y2)

    visible_area = max(0, vx2 - vx1) * max(0, vy2 - vy1)

    return visible_area / full_area


def blur_value(face_img):
    if face_img is None or face_img.size == 0:
        return 0.0

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def score_size(bbox):
    x1, y1, x2, y2 = bbox
    face_w = x2 - x1
    face_h = y2 - y1

    min_side = min(face_w, face_h)

    return max(0.0, min(1.0, min_side / GOOD_FACE_SIZE))


def score_blur(face_img):
    val = blur_value(face_img)
    score = max(0.0, min(1.0, val / GOOD_BLUR))

    return score, val


def score_visibility(img, bbox):
    return max(0.0, min(1.0, visible_bbox_ratio(img, bbox)))


def score_aspect(bbox):
    x1, y1, x2, y2 = bbox
    face_w = x2 - x1
    face_h = y2 - y1

    if face_h <= 0:
        return 0.0, None

    ratio = face_w / face_h

    if IDEAL_ASPECT_MIN <= ratio <= IDEAL_ASPECT_MAX:
        return 1.0, ratio

    if ratio < MIN_ASPECT or ratio > MAX_ASPECT:
        return 0.0, ratio

    if ratio < IDEAL_ASPECT_MIN:
        score = (ratio - MIN_ASPECT) / (IDEAL_ASPECT_MIN - MIN_ASPECT)
    else:
        score = (MAX_ASPECT - ratio) / (MAX_ASPECT - IDEAL_ASPECT_MAX)

    return max(0.0, min(1.0, score)), ratio


def score_landmarks(face):
    kps = getattr(face, "kps", None)

    if kps is None or len(kps) < 5:
        return 0.0, {
            "has_kps": False,
        }

    left_eye, right_eye, nose, left_mouth, right_mouth = kps

    eye_dist = math.dist(left_eye, right_eye)
    mouth_dist = math.dist(left_mouth, right_mouth)

    points = 0
    total = 4

    if eye_dist >= MIN_EYE_DIST:
        points += 1

    if mouth_dist >= MIN_MOUTH_DIST:
        points += 1

    if nose[1] > min(left_eye[1], right_eye[1]):
        points += 1

    if max(left_mouth[1], right_mouth[1]) > nose[1]:
        points += 1

    score = points / total

    return score, {
        "has_kps": True,
        "eye_dist": float(eye_dist),
        "mouth_dist": float(mouth_dist),
        "points": points,
        "total": total,
    }


def get_face_quality(img, face):
    bbox = face.bbox.astype(int)

    face_img = crop_face(img, bbox)

    if face_img is None:
        return 0.0, {
            "reason": "bad_crop",
            "bbox": bbox.tolist(),
            "det_score": float(getattr(face, "det_score", 0.0) or 0.0),
        }

    q_size = score_size(bbox)
    q_blur, blur = score_blur(face_img)
    q_visibility = score_visibility(img, face.bbox)
    q_aspect, aspect_ratio = score_aspect(bbox)
    q_landmarks, landmarks_details = score_landmarks(face)

    quality = (
        0.35 * q_size
        + 0.10 * q_blur
        + 0.35 * q_visibility
        + 0.10 * q_aspect
        + 0.10 * q_landmarks
    )

    details = {
        "bbox": bbox.tolist(),
        "det_score": float(getattr(face, "det_score", 0.0) or 0.0),
        "size_score": round(q_size, 4),
        "blur_score": round(q_blur, 4),
        "blur_value": round(float(blur), 4),
        "visibility_score": round(q_visibility, 4),
        "aspect_score": round(q_aspect, 4),
        "aspect_ratio": round(float(aspect_ratio), 4) if aspect_ratio is not None else None,
        "landmarks_score": round(q_landmarks, 4),
        "landmarks": landmarks_details,
    }

    return round(float(quality), 4), details
