import cv2


MIN_FACE_SIZE = 55
BLUR_THRESHOLD = 25

MIN_ASPECT = 0.55
MAX_ASPECT = 1.55

MIN_DET_SCORE = 0.50
VISIBLE_FACE_THRESHOLD = 0.70

MIN_EYE_DIST = 4
MIN_MOUTH_DIST = 2


def crop_face(img, bbox):
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def visible_bbox_ratio(img, bbox):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)
    full_area = max(0, x2 - x1) * max(0, y2 - y1)
    if full_area <= 0:
        return 0.0
    vx1, vy1, vx2, vy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    visible_area = max(0, vx2 - vx1) * max(0, vy2 - vy1)
    return visible_area / full_area


def is_blurry(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_value < BLUR_THRESHOLD


def has_good_aspect_ratio(bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if h <= 0:
        return False
    ratio = w / h
    return MIN_ASPECT <= ratio <= MAX_ASPECT


def landmarks_are_usable(face):
    kps = getattr(face, "kps", None)

    if kps is None or len(kps) < 5:
        return False

    left_eye, right_eye, nose, left_mouth, right_mouth = kps

    eye_dist = abs(right_eye[0] - left_eye[0])
    mouth_dist = abs(right_mouth[0] - left_mouth[0])

    if eye_dist < MIN_EYE_DIST:
        return False

    if mouth_dist < MIN_MOUTH_DIST:
        return False


    return True


def is_good_face(img, face):
    bbox = face.bbox.astype(int)

    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]

    # 1. Надто мале обличчя
    if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
        return False

    # 2. Низька впевненість детектора
    if float(face.det_score) < MIN_DET_SCORE:
        return False

    # 3. Обрізане обличчя / видно менше 70%
    if visible_bbox_ratio(img, face.bbox) < VISIBLE_FACE_THRESHOLD:
        return False

    # 4. Неправильна форма bbox
    if not has_good_aspect_ratio(bbox):
        return False

    face_img = crop_face(img, bbox)
    if face_img is None:
        return False

    # 5. Розмите
    if is_blurry(face_img):
        return False

    # 6. Погані landmarks / сильний поворот / фрагмент обличчя
    if not landmarks_are_usable(face):
        return False

    return True
