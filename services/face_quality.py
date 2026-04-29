import cv2


MIN_FACE_SIZE = 40        # мінімальний розмір (px)
GOOD_FACE_SIZE = 160      # розмір, який вважаємо ідеальним
BLUR_LOW = 50             # дуже розмите
BLUR_HIGH = 200           # чітке
MIN_ASPECT = 0.6   # занадто вузьке (профіль / обрізане)
MAX_ASPECT = 1.6   # занадто широке

QUALITY_THRESHOLD = 0.55  # поріг прийняття


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


def norm_det_score(det_score):
    return float(det_score)


def norm_size(bbox):
    x1, y1, x2, y2 = bbox
    size = min(x2 - x1, y2 - y1)

    if size <= MIN_FACE_SIZE:
        return 0.0
    if size >= GOOD_FACE_SIZE:
        return 1.0

    return (size - MIN_FACE_SIZE) / (GOOD_FACE_SIZE - MIN_FACE_SIZE)


def norm_blur(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    val = cv2.Laplacian(gray, cv2.CV_64F).var()

    if val <= BLUR_LOW:
        return 0.0
    if val >= BLUR_HIGH:
        return 1.0

    return (val - BLUR_LOW) / (BLUR_HIGH - BLUR_LOW)


def norm_aspect(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    if h == 0:
        return 0.0

    ratio = w / h

    if ratio < MIN_ASPECT or ratio > MAX_ASPECT:
        return 0.0

    # ідеал ~1.0
    return 1.0 - abs(ratio - 1.0)


def is_good_face(img, face):
    bbox = face.bbox.astype(int)

    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]

    if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
        return 0.0

    face_img = crop_face(img, bbox)
    if face_img is None:
        return 0.0

    q_det = norm_det_score(face.det_score)
    q_size = norm_size(bbox)
    q_blur = norm_blur(face_img)
    q_aspect = norm_aspect(bbox)

    score = (0.35 * q_det + 0.25 * q_size + 0.25 * q_blur + 0.15 * q_aspect)


    return float(score)


def pass_quality(img, face):
    return is_good_face(img, face) >= QUALITY_THRESHOLD
