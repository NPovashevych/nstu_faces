import csv
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import faiss
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from sqlalchemy.orm import Session

from db.session import SessionLocal
from db.enums import EmbeddingType, FaceGender, PersonStatus
from db.models import (
    DBEmbedding,
    DBFaceCategory,
    DBFreeze,
    DBMedia,
    DBPerson,
)

from routes.routers_classic.commons import normalize

from services.create_faces.face_quality_v3 import get_face_quality
from services.create_faces.clip_face_filter_v2 import (
    get_clip,
    analyze_face_category,
)
from services.create_faces.clip_face_categories import (
    DEFAULT_FACE_CATEGORY,
    CATEGORY_IDENTIFIABLE,
    CATEGORY_LOW_QUALITY,
)


# ============================================================
# TEST CONFIG
# ============================================================

SERVICE_NAME = "detect_faces_from_freezes_v5_test.py"

TEST_SOURCE_IDS = (1, 5)
CONTROL_CSV = Path(r"C:\pg_statistic\data-1786619345681.csv")
V5_RESULT_CSV = Path(r"C:\pg_statistic\face_v5_test.csv")
LOG_FILE = Path(r"../logs/detect_faces_from_freezes_v5_test.log")

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
V5_RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# PARAMETERS — ті самі, що у v4
# ============================================================

FACE_DET_SIZE = 640

MIN_DET_SCORE = 0.60

DIST_TOLERANCE = 0.45
STEP_TOLERANCE = 0.055

UNKNOWN_TOLERANCE = 0.55

LOW_QUALITY_THRESHOLD = 0.60

# Для зіставлення old/new detection.
BBOX_IOU_THRESHOLD = 0.95

QUALITY_COMPARE_TOLERANCE = 0.0001
CATEGORY_SCORE_COMPARE_TOLERANCE = 0.0001

# Щоб при великій кількості розбіжностей не засипати консоль.
# Повний результат v5 все одно буде в CSV.
DIFF_LOG_LIMIT = 500

PROGRESS_EVERY_MEDIA = 100


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        ),
    ],
)


# ============================================================
# INSIGHTFACE
# ============================================================

_INSIGHTFACE_CACHE = None


def load_insightface():
    app = FaceAnalysis(
        name="buffalo_l",
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    app.prepare(
        ctx_id=0,
        det_size=(FACE_DET_SIZE, FACE_DET_SIZE),
    )

    return app


def get_insightface():
    global _INSIGHTFACE_CACHE

    if _INSIGHTFACE_CACHE is None:
        _INSIGHTFACE_CACHE = load_insightface()

    return _INSIGHTFACE_CACHE


# ============================================================
# HELPERS
# ============================================================

def enum_value(value):
    if value is None:
        return None

    if hasattr(value, "value"):
        return value.value

    return str(value)


def normalize_vector(vector) -> np.ndarray:
    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def optional_float(value):
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(result):
        return None

    return result


def floats_equal(a, b, tolerance):
    a = optional_float(a)
    b = optional_float(b)

    if a is None and b is None:
        return True

    if a is None or b is None:
        return False

    return abs(a - b) <= tolerance


# ============================================================
# FACE CATEGORIES
# ============================================================

def load_face_categories(db: Session):
    rows = (
        db.query(DBFaceCategory)
        .filter(DBFaceCategory.is_active.is_(True))
        .all()
    )

    categories = {
        row.name: row
        for row in rows
    }

    logging.info(
        f"Loaded active face categories: {len(categories)}"
    )

    return categories


def get_category(
    face_categories: dict,
    category_name: str,
) -> DBFaceCategory:

    category = face_categories.get(category_name)

    if category:
        return category

    fallback = face_categories.get(
        DEFAULT_FACE_CATEGORY
    )

    if fallback:
        logging.warning(
            f"Face category '{category_name}' not found. "
            f"Fallback to '{DEFAULT_FACE_CATEGORY}'."
        )

        return fallback

    raise RuntimeError(
        f"Face category '{category_name}' not found, "
        f"fallback '{DEFAULT_FACE_CATEGORY}' not found."
    )


# ============================================================
# PERSON LOOKUP — READ ONLY
# ============================================================

def load_person_lookup(db: Session):
    rows = db.query(DBPerson).all()

    result = {}

    for person in rows:
        result[person.id] = {
            "id": person.id,
            "name": person.name,
            "code": person.code,
            "status": enum_value(person.status),
        }

    logging.info(
        f"Loaded persons for lookup: {len(result)}"
    )

    return result


def get_service_person(
    db: Session,
    category_name: str,
):
    """
    READ ONLY.

    У production v5 тут можна буде створювати службову
    person при її відсутності.

    У тесті БД не змінюємо.
    """

    cluster_tag = f"{category_name}_cluster"

    return (
        db.query(DBPerson)
        .filter(DBPerson.code == cluster_tag)
        .first()
    )


# ============================================================
# CONFIDENCE
# ============================================================

def get_confidence(dist: float) -> int:
    if dist <= DIST_TOLERANCE:
        return 0

    for i in range(1, 4):
        if (
            dist
            <= DIST_TOLERANCE
            + i * STEP_TOLERANCE
        ):
            return i

    return -1


# ============================================================
# REFERENCE FAISS
# ============================================================

class ReferenceIndex:
    """
    Exact FAISS index для reference_face.

    PostgreSQL є source of truth.
    Індекс будується заново при старті тесту.
    """

    def __init__(self):
        self.index = None
        self.metadata = []
        self.dimension = None

    def build(self, db: Session):
        rows = (
            db.query(DBEmbedding)
            .filter(
                DBEmbedding.embedding_type
                == EmbeddingType.reference_face
            )
            .order_by(DBEmbedding.id)
            .all()
        )

        if not rows:
            logging.warning(
                "Reference embeddings not found."
            )
            return

        vectors = []
        metadata = []

        for row in rows:
            vector = normalize_vector(
                row.vector
            )

            vectors.append(vector)

            metadata.append(
                {
                    "embedding_id": row.id,
                    "person_id": row.person_id,
                }
            )

        matrix = np.vstack(
            vectors
        ).astype(np.float32)

        self.dimension = matrix.shape[1]

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.index.add(matrix)

        self.metadata = metadata

        logging.info(
            "Reference FAISS built: "
            f"{self.index.ntotal} embeddings, "
            f"dimension={self.dimension}"
        )

    def find(self, embedding):
        if (
            self.index is None
            or self.index.ntotal == 0
        ):
            return None, 1.0

        vector = normalize_vector(
            embedding
        ).reshape(1, -1)

        similarities, indices = (
            self.index.search(
                vector,
                1,
            )
        )

        index_position = int(
            indices[0][0]
        )

        if index_position < 0:
            return None, 1.0

        similarity = float(
            similarities[0][0]
        )

        distance = 1.0 - similarity

        metadata = self.metadata[
            index_position
        ]

        return metadata, distance


# ============================================================
# UNKNOWN FAISS — TEST ONLY
# ============================================================

class UnknownTestIndex:
    """
    Динамічний unknown index.

    ВАЖЛИВО:
    він стартує ПОРОЖНІМ.

    Старі detected_face з PostgreSQL сюди НЕ завантажуються,
    бо вони є результатом нашого контрольного набору і
    підглядали б у майбутнє.

    Усі нові кластери живуть тільки в RAM.
    """

    def __init__(self, dimension=512):
        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

        # metadata[position] -> cluster_tag
        self.metadata = []

        self.next_cluster_id = 1

    def create_cluster(self):
        cluster_tag = (
            f"test_unknown_"
            f"{self.next_cluster_id:06d}"
        )

        self.next_cluster_id += 1

        return cluster_tag

    def find(self, embedding):
        if self.index.ntotal == 0:
            return None, None

        vector = normalize_vector(
            embedding
        ).reshape(1, -1)

        similarities, indices = (
            self.index.search(
                vector,
                1,
            )
        )

        position = int(
            indices[0][0]
        )

        if position < 0:
            return None, None

        similarity = float(
            similarities[0][0]
        )

        distance = 1.0 - similarity

        cluster_tag = self.metadata[
            position
        ]

        return cluster_tag, distance

    def add(
        self,
        embedding,
        cluster_tag: str,
    ):
        vector = normalize_vector(
            embedding
        ).reshape(1, -1)

        self.index.add(vector)

        self.metadata.append(
            cluster_tag
        )

    def find_or_create(
        self,
        embedding,
    ):
        cluster_tag, distance = (
            self.find(embedding)
        )

        if (
            cluster_tag is not None
            and distance is not None
            and distance <= UNKNOWN_TOLERANCE
        ):
            self.add(
                embedding,
                cluster_tag,
            )

            return (
                cluster_tag,
                distance,
                False,
            )

        cluster_tag = self.create_cluster()

        self.add(
            embedding,
            cluster_tag,
        )

        return (
            cluster_tag,
            None,
            True,
        )


# ============================================================
# ANALYSIS
# ============================================================

def make_analysis(
    quality_details: dict,
    category_result: dict,
):
    return {
        "quality": quality_details,
        "clip": {
            "category":
                category_result["category"],

            "category_score":
                category_result["category_score"],

            "best_clip_category":
                category_result["best_clip_category"],

            "best_clip_score":
                category_result["best_clip_score"],

            "clip_scores":
                category_result["clip_scores"],
        },
    }


def make_analysis_without_clip(
    quality_details: dict,
    reason: str,
    category_name: str,
):
    return {
        "quality": quality_details,
        "clip": {
            "category": category_name,
            "category_score": None,
            "best_clip_category": reason,
            "best_clip_score": None,
            "clip_scores": None,
        },
    }


# ============================================================
# GENDER
# ============================================================

def map_gender(
    face,
    category_name: str,
):
    if category_name != CATEGORY_IDENTIFIABLE:
        return FaceGender.unknown

    gender = getattr(
        face,
        "gender",
        None,
    )

    if gender == 1:
        return FaceGender.male

    if gender == 0:
        return FaceGender.female

    return FaceGender.unknown


# ============================================================
# TEST MEDIA
# ============================================================

def get_test_media(db: Session):
    medias = (
        db.query(DBMedia)
        .join(
            DBFreeze,
            DBFreeze.media_id == DBMedia.id,
        )
        .filter(
            DBMedia.source_id.in_(
                TEST_SOURCE_IDS
            )
        )
        .distinct()
        .order_by(DBMedia.id)
        .all()
    )

    logging.info(
        f"TEST media source_id={TEST_SOURCE_IDS}: "
        f"{len(medias)}"
    )

    return medias


# ============================================================
# RESULT ROW
# ============================================================

def make_result_row(
    media: DBMedia,
    freeze: DBFreeze,
    face,
    det_score,
    quality,
    gender,
    confidence,
    category,
    category_score,
    analysis,
    result_type,
    person_id=None,
    person_name=None,
    person_status=None,
    reference_embedding_id=None,
    reference_distance=None,
    unknown_cluster_tag=None,
    unknown_distance=None,
):

    bbox = (
        face.bbox
        .astype(float)
        .tolist()
    )

    return {
        "media_id": media.id,
        "source_id": media.source_id,

        "freeze_id": freeze.id,
        "freeze_path": freeze.freeze_path,

        "bbox": json.dumps(
            bbox,
            ensure_ascii=False,
        ),

        "det_score": det_score,
        "quality": quality,

        "gender": enum_value(
            gender
        ),

        "confidence": confidence,

        "category_id": category.id,
        "category": category.name,
        "category_score": category_score,

        "result": result_type,

        "person_id": person_id,
        "person_name": person_name,
        "person_status": person_status,

        "reference_embedding_id":
            reference_embedding_id,

        "reference_distance":
            reference_distance,

        "unknown_cluster_tag":
            unknown_cluster_tag,

        "unknown_distance":
            unknown_distance,

        "analysis": json.dumps(
            analysis,
            ensure_ascii=False,
        ),
    }


# ============================================================
# PROCESS FREEZE — NO DB WRITES
# ============================================================

def process_freeze(
    db: Session,
    face_model,
    clip_model,
    clip_preprocess,
    clip_text_features,
    clip_prompt_categories,
    media: DBMedia,
    freeze: DBFreeze,
    reference_index: ReferenceIndex,
    unknown_index: UnknownTestIndex,
    face_categories: dict,
    person_lookup: dict,
    stats: dict,
):

    img = cv2.imread(
        freeze.freeze_path
    )

    if img is None:
        stats["cannot_read"] += 1

        logging.warning(
            f"Cannot read freeze: "
            f"{freeze.freeze_path}"
        )

        return []

    faces = face_model.get(img)

    stats["freezes"] += 1
    stats["buffalo_faces"] += len(faces)

    if not faces:
        stats["freezes_without_faces"] += 1
        return []

    results = []

    # PIL відкриваємо тільки якщо реально знадобиться CLIP.
    pil_image = None

    for face in faces:
        bbox = (
            face.bbox
            .astype(float)
            .tolist()
        )

        det_score = float(
            getattr(
                face,
                "det_score",
                0.0,
            ) or 0.0
        )

        emb = normalize_vector(
            face.embedding
        )

        # ====================================================
        # 1. KNOWN — FAISS
        # ====================================================

        best_ref, best_dist = (
            reference_index.find(
                emb
            )
        )

        confidence = get_confidence(
            best_dist
        )

        can_use_known_match = (
            best_ref is not None
            and confidence != -1
        )

        if can_use_known_match:
            category = get_category(
                face_categories,
                CATEGORY_IDENTIFIABLE,
            )

            category_score = confidence

            quality, quality_details = (
                get_face_quality(
                    img,
                    face,
                )
            )

            analysis = (
                make_analysis_without_clip(
                    quality_details=
                        quality_details,

                    reason=
                        "not_checked_known_match",

                    category_name=
                        category.name,
                )
            )

            person_id = (
                best_ref["person_id"]
            )

            person_data = (
                person_lookup.get(
                    person_id,
                    {},
                )
            )

            distance_for_source = round(
                best_dist,
                4,
            )

            results.append(
                make_result_row(
                    media=media,
                    freeze=freeze,
                    face=face,

                    det_score=det_score,
                    quality=quality,

                    gender=map_gender(
                        face,
                        category.name,
                    ),

                    confidence=confidence,

                    category=category,
                    category_score=
                        category_score,

                    analysis=analysis,

                    result_type="known",

                    person_id=person_id,

                    person_name=
                        person_data.get(
                            "name"
                        ),

                    person_status=
                        person_data.get(
                            "status"
                        ),

                    reference_embedding_id=
                        best_ref[
                            "embedding_id"
                        ],

                    reference_distance=
                        distance_for_source,
                )
            )

            stats["known"] += 1

            continue

        # ====================================================
        # 2. LOW DET SCORE
        # ====================================================

        if det_score < MIN_DET_SCORE:
            stats["skipped_low_det_score"] += 1

            logging.debug(
                f"Skip low det_score "
                f"freeze_id={freeze.id}, "
                f"det_score={det_score}, "
                f"bbox={bbox}"
            )

            continue

        # ====================================================
        # 3. QUALITY
        # ====================================================

        quality, quality_details = (
            get_face_quality(
                img,
                face,
            )
        )

        if quality < LOW_QUALITY_THRESHOLD:
            category = get_category(
                face_categories,
                CATEGORY_LOW_QUALITY,
            )

            category_score = quality

            analysis = (
                make_analysis_without_clip(
                    quality_details=
                        quality_details,

                    reason=
                        "not_checked_low_quality",

                    category_name=
                        category.name,
                )
            )

            service_person = (
                get_service_person(
                    db,
                    category.name,
                )
            )

            results.append(
                make_result_row(
                    media=media,
                    freeze=freeze,
                    face=face,

                    det_score=det_score,
                    quality=quality,

                    gender=
                        FaceGender.unknown,

                    confidence=None,

                    category=category,
                    category_score=
                        category_score,

                    analysis=analysis,

                    result_type=
                        "low_quality",

                    person_id=(
                        service_person.id
                        if service_person
                        else None
                    ),

                    person_name=(
                        service_person.name
                        if service_person
                        else None
                    ),

                    person_status=(
                        enum_value(
                            service_person.status
                        )
                        if service_person
                        else "suspicious"
                    ),
                )
            )

            stats["low_quality"] += 1

            continue

        # ====================================================
        # 4. CLIP
        # ====================================================

        if pil_image is None:
            try:
                pil_image = (
                    Image.open(
                        freeze.freeze_path
                    )
                    .convert("RGB")
                )

            except Exception as e:
                stats["cannot_open_pil"] += 1

                logging.warning(
                    f"Cannot open PIL: "
                    f"{freeze.freeze_path} | {e}"
                )

                continue

        category_result = (
            analyze_face_category(
                image=pil_image,
                bbox=bbox,

                model=clip_model,
                preprocess=clip_preprocess,

                text_features=
                    clip_text_features,

                prompt_categories=
                    clip_prompt_categories,
            )
        )

        category_name = (
            category_result.get(
                "category"
            )
            or DEFAULT_FACE_CATEGORY
        )

        category = get_category(
            face_categories,
            category_name,
        )

        category_score = (
            category_result[
                "category_score"
            ]
        )

        analysis = make_analysis(
            quality_details=
                quality_details,

            category_result=
                category_result,
        )

        # ====================================================
        # 5. IDENTIFIABLE UNKNOWN — FAISS IN RAM
        # ====================================================

        if (
            category.name
            == CATEGORY_IDENTIFIABLE
        ):
            (
                cluster_tag,
                cluster_dist,
                is_new_cluster,
            ) = unknown_index.find_or_create(
                emb
            )

            distance_for_source = (
                round(
                    cluster_dist,
                    4,
                )
                if cluster_dist is not None
                else None
            )

            result_type = (
                "unknown_new"
                if is_new_cluster
                else "unknown_existing"
            )

            results.append(
                make_result_row(
                    media=media,
                    freeze=freeze,
                    face=face,

                    det_score=det_score,
                    quality=quality,

                    gender=map_gender(
                        face,
                        category.name,
                    ),

                    confidence=None,

                    category=category,
                    category_score=
                        category_score,

                    analysis=analysis,

                    result_type=
                        result_type,

                    person_id=None,
                    person_name=
                        cluster_tag,

                    person_status=
                        enum_value(
                            PersonStatus.unknown
                        ),

                    unknown_cluster_tag=
                        cluster_tag,

                    unknown_distance=
                        distance_for_source,
                )
            )

            if is_new_cluster:
                stats[
                    "unknown_new"
                ] += 1
            else:
                stats[
                    "unknown_existing"
                ] += 1

            continue

        # ====================================================
        # 6. SERVICE CATEGORY
        # ====================================================

        service_person = (
            get_service_person(
                db,
                category.name,
            )
        )

        results.append(
            make_result_row(
                media=media,
                freeze=freeze,
                face=face,

                det_score=det_score,
                quality=quality,

                gender=
                    FaceGender.unknown,

                confidence=None,

                category=category,
                category_score=
                    category_score,

                analysis=analysis,

                result_type=
                    "service_category",

                person_id=(
                    service_person.id
                    if service_person
                    else None
                ),

                person_name=(
                    service_person.name
                    if service_person
                    else
                    f"{category.name}_cluster"
                ),

                person_status=(
                    enum_value(
                        service_person.status
                    )
                    if service_person
                    else "suspicious"
                ),
            )
        )

        stats["service_category"] += 1

    return results


# ============================================================
# SAVE V5 RESULT
# ============================================================

def save_results_csv(
    results,
    output_path: Path,
):
    if not results:
        logging.warning(
            "No results to save."
        )
        return

    fieldnames = list(
        results[0].keys()
    )

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    logging.info(
        f"V5 result saved: "
        f"{output_path}"
    )


# ============================================================
# OLD CSV PARSING
# ============================================================

def parse_bbox(value):
    if value is None:
        return None

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            float(x)
            for x in value
        ]

    text = str(value).strip()

    if not text:
        return None

    # PostgreSQL array:
    # {1.2,3.4,5.6,7.8}
    if (
        text.startswith("{")
        and text.endswith("}")
    ):
        text = text[1:-1]

        return [
            float(x.strip())
            for x in text.split(",")
        ]

    # JSON:
    # [1.2, 3.4, 5.6, 7.8]
    if (
        text.startswith("[")
        and text.endswith("]")
    ):
        return [
            float(x)
            for x in json.loads(text)
        ]

    raise ValueError(
        f"Unknown bbox format: {value}"
    )


def load_control_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"CONTROL_CSV not found: {path}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            row["_bbox"] = parse_bbox(
                row["bbox"]
            )

            row["freeze_id"] = int(
                row["freeze_id"]
            )

            row["person_id"] = int(
                row["person_id"]
            )

            row["category_id"] = int(
                row["category_id"]
            )

            row["confidence"] = (
                optional_float(
                    row["confidence"]
                )
            )

            row["quality"] = (
                optional_float(
                    row["quality"]
                )
            )

            row["category_score"] = (
                optional_float(
                    row["category_score"]
                )
            )

            rows.append(row)

    logging.info(
        f"Control CSV loaded: "
        f"{len(rows)} rows"
    )

    return rows


# ============================================================
# IOU
# ============================================================

def bbox_iou(a, b):
    if (
        a is None
        or b is None
    ):
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = iw * ih

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# CLASSIFY OLD RESULT
# ============================================================

def old_result_class(
    old_row,
    person_lookup,
    category_by_id,
):
    confidence = old_row[
        "confidence"
    ]

    if confidence is not None:
        return "known"

    person = person_lookup.get(
        old_row["person_id"],
        {},
    )

    status = person.get(
        "status"
    )

    category = category_by_id.get(
        old_row["category_id"]
    )

    category_name = (
        category.name
        if category
        else None
    )

    if (
        category_name
        == CATEGORY_LOW_QUALITY
    ):
        return "low_quality"

    if (
        status
        == enum_value(
            PersonStatus.unknown
        )
        and category_name
        == CATEGORY_IDENTIFIABLE
    ):
        return "unknown"

    return "service_category"


def new_result_class(
    new_row,
):
    result = new_row["result"]

    if result.startswith(
        "unknown_"
    ):
        return "unknown"

    return result


# ============================================================
# MATCH OLD / NEW BY FREEZE + IOU
# ============================================================

def match_faces(
    old_rows,
    new_rows,
):
    old_by_freeze = defaultdict(list)
    new_by_freeze = defaultdict(list)

    for row in old_rows:
        old_by_freeze[
            row["freeze_id"]
        ].append(row)

    for row in new_rows:
        row["_bbox"] = parse_bbox(
            row["bbox"]
        )

        new_by_freeze[
            int(row["freeze_id"])
        ].append(row)

    matched = []
    only_old = []
    only_new = []

    all_freeze_ids = (
        set(old_by_freeze)
        | set(new_by_freeze)
    )

    for freeze_id in all_freeze_ids:
        old_group = old_by_freeze.get(
            freeze_id,
            [],
        )

        new_group = new_by_freeze.get(
            freeze_id,
            [],
        )

        candidates = []

        for old_index, old_row in enumerate(
            old_group
        ):
            for new_index, new_row in enumerate(
                new_group
            ):
                iou = bbox_iou(
                    old_row["_bbox"],
                    new_row["_bbox"],
                )

                if (
                    iou
                    >= BBOX_IOU_THRESHOLD
                ):
                    candidates.append(
                        (
                            iou,
                            old_index,
                            new_index,
                        )
                    )

        # Спочатку найкращі IoU.
        candidates.sort(
            reverse=True,
            key=lambda x: x[0],
        )

        used_old = set()
        used_new = set()

        for (
            iou,
            old_index,
            new_index,
        ) in candidates:

            if old_index in used_old:
                continue

            if new_index in used_new:
                continue

            used_old.add(old_index)
            used_new.add(new_index)

            matched.append(
                (
                    old_group[old_index],
                    new_group[new_index],
                    iou,
                )
            )

        for index, row in enumerate(
            old_group
        ):
            if index not in used_old:
                only_old.append(row)

        for index, row in enumerate(
            new_group
        ):
            if index not in used_new:
                only_new.append(row)

    return (
        matched,
        only_old,
        only_new,
    )


# ============================================================
# COMPARISON
# ============================================================

def compare_results(
    db: Session,
    old_rows,
    new_rows,
    person_lookup,
    face_categories,
):

    category_by_id = {
        category.id: category
        for category
        in face_categories.values()
    }

    (
        matched,
        only_old,
        only_new,
    ) = match_faces(
        old_rows,
        new_rows,
    )

    same_category = 0
    different_category = 0

    same_confidence = 0
    different_confidence = 0

    same_quality = 0
    different_quality = 0

    same_category_score = 0
    different_category_score = 0

    same_result_class = 0
    different_result_class = 0

    same_known_person = 0
    different_known_person = 0

    old_unknown_new_known = 0
    old_known_new_unknown = 0

    matched_unknown = 0

    # Для аналізу clustering.
    old_to_new_clusters = defaultdict(
        set
    )

    new_to_old_clusters = defaultdict(
        set
    )

    diffs = []

    for old_row, new_row, iou in matched:

        old_class = old_result_class(
            old_row,
            person_lookup,
            category_by_id,
        )

        new_class = new_result_class(
            new_row
        )

        row_diff = []

        # --------------------------------
        # result class
        # --------------------------------

        if old_class == new_class:
            same_result_class += 1
        else:
            different_result_class += 1

            row_diff.append(
                f"class "
                f"{old_class} -> {new_class}"
            )

        # --------------------------------
        # category
        # --------------------------------

        if (
            int(old_row["category_id"])
            == int(new_row["category_id"])
        ):
            same_category += 1
        else:
            different_category += 1

            row_diff.append(
                "category_id "
                f"{old_row['category_id']} "
                f"-> {new_row['category_id']}"
            )

        # --------------------------------
        # confidence
        # --------------------------------

        if floats_equal(
            old_row["confidence"],
            new_row["confidence"],
            0.000001,
        ):
            same_confidence += 1
        else:
            different_confidence += 1

            row_diff.append(
                "confidence "
                f"{old_row['confidence']} "
                f"-> {new_row['confidence']}"
            )

        # --------------------------------
        # quality
        # --------------------------------

        if floats_equal(
            old_row["quality"],
            new_row["quality"],
            QUALITY_COMPARE_TOLERANCE,
        ):
            same_quality += 1
        else:
            different_quality += 1

        # --------------------------------
        # category_score
        # --------------------------------

        if floats_equal(
            old_row["category_score"],
            new_row["category_score"],
            CATEGORY_SCORE_COMPARE_TOLERANCE,
        ):
            same_category_score += 1
        else:
            different_category_score += 1

        # --------------------------------
        # KNOWN person comparison
        # --------------------------------

        if (
            old_class == "known"
            and new_class == "known"
        ):
            old_person_id = int(
                old_row["person_id"]
            )

            new_person_id = (
                int(new_row["person_id"])
                if new_row["person_id"]
                not in (None, "")
                else None
            )

            if (
                old_person_id
                == new_person_id
            ):
                same_known_person += 1

            else:
                different_known_person += 1

                row_diff.append(
                    "known person "
                    f"{old_person_id} "
                    f"-> {new_person_id}"
                )

        # --------------------------------
        # IMPORTANT:
        # old unknown -> new known
        # --------------------------------

        if (
            old_class == "unknown"
            and new_class == "known"
        ):
            old_unknown_new_known += 1

            row_diff.append(
                "OLD UNKNOWN -> NEW KNOWN "
                f"person_id="
                f"{new_row['person_id']}"
            )

        if (
            old_class == "known"
            and new_class == "unknown"
        ):
            old_known_new_unknown += 1

            row_diff.append(
                "OLD KNOWN -> NEW UNKNOWN"
            )

        # --------------------------------
        # UNKNOWN clustering comparison
        # --------------------------------

        if (
            old_class == "unknown"
            and new_class == "unknown"
        ):
            matched_unknown += 1

            old_person_id = int(
                old_row["person_id"]
            )

            new_cluster_tag = (
                new_row[
                    "unknown_cluster_tag"
                ]
            )

            old_to_new_clusters[
                old_person_id
            ].add(
                new_cluster_tag
            )

            new_to_old_clusters[
                new_cluster_tag
            ].add(
                old_person_id
            )

        if row_diff:
            diffs.append(
                {
                    "freeze_id":
                        old_row["freeze_id"],

                    "iou": iou,

                    "old_bbox":
                        old_row["_bbox"],

                    "new_bbox":
                        new_row["_bbox"],

                    "old_person_id":
                        old_row["person_id"],

                    "new_person_id":
                        new_row["person_id"],

                    "new_cluster":
                        new_row[
                            "unknown_cluster_tag"
                        ],

                    "details":
                        row_diff,
                }
            )

    # ========================================================
    # UNKNOWN SPLITS / MERGES
    # ========================================================

    split_old_clusters = {
        person_id: clusters
        for person_id, clusters
        in old_to_new_clusters.items()
        if len(clusters) > 1
    }

    merged_new_clusters = {
        cluster_tag: person_ids
        for cluster_tag, person_ids
        in new_to_old_clusters.items()
        if len(person_ids) > 1
    }

    # ========================================================
    # SUMMARY
    # ========================================================

    logging.info("")
    logging.info(
        "=" * 70
    )
    logging.info(
        "V4 -> V5 REGRESSION TEST"
    )
    logging.info(
        "=" * 70
    )

    logging.info(
        f"OLD faces:               "
        f"{len(old_rows)}"
    )

    logging.info(
        f"NEW faces:               "
        f"{len(new_rows)}"
    )

    logging.info("")

    logging.info(
        f"Matched detections:       "
        f"{len(matched)}"
    )

    logging.info(
        f"Only OLD:                 "
        f"{len(only_old)}"
    )

    logging.info(
        f"Only NEW:                 "
        f"{len(only_new)}"
    )

    logging.info("")

    logging.info(
        f"Same result class:        "
        f"{same_result_class}"
    )

    logging.info(
        f"Different result class:   "
        f"{different_result_class}"
    )

    logging.info("")

    logging.info(
        f"Same category_id:         "
        f"{same_category}"
    )

    logging.info(
        f"Different category_id:    "
        f"{different_category}"
    )

    logging.info("")

    logging.info(
        f"Same confidence:          "
        f"{same_confidence}"
    )

    logging.info(
        f"Different confidence:     "
        f"{different_confidence}"
    )

    logging.info("")

    logging.info(
        f"Same quality:             "
        f"{same_quality}"
    )

    logging.info(
        f"Different quality:        "
        f"{different_quality}"
    )

    logging.info("")

    logging.info(
        f"Same category_score:      "
        f"{same_category_score}"
    )

    logging.info(
        f"Different category_score: "
        f"{different_category_score}"
    )

    logging.info("")

    logging.info(
        f"Same KNOWN person_id:     "
        f"{same_known_person}"
    )

    logging.info(
        f"Different KNOWN person:   "
        f"{different_known_person}"
    )

    logging.info("")

    logging.info(
        f"OLD unknown -> NEW known: "
        f"{old_unknown_new_known}"
    )

    logging.info(
        f"OLD known -> NEW unknown: "
        f"{old_known_new_unknown}"
    )

    logging.info("")

    logging.info(
        f"Matched UNKNOWN faces:    "
        f"{matched_unknown}"
    )

    logging.info(
        f"Old unknown clusters:     "
        f"{len(old_to_new_clusters)}"
    )

    logging.info(
        f"New unknown clusters:     "
        f"{len(new_to_old_clusters)}"
    )

    logging.info(
        f"Old clusters SPLIT:       "
        f"{len(split_old_clusters)}"
    )

    logging.info(
        f"New clusters MERGED:      "
        f"{len(merged_new_clusters)}"
    )

    logging.info(
        "=" * 70
    )

    # ========================================================
    # ONLY OLD
    # ========================================================

    if only_old:
        logging.info("")
        logging.info(
            "--- ONLY OLD DETECTIONS ---"
        )

        for row in only_old[
            :DIFF_LOG_LIMIT
        ]:
            logging.info(
                f"ONLY OLD | "
                f"freeze_id={row['freeze_id']} | "
                f"person_id={row['person_id']} | "
                f"category_id={row['category_id']} | "
                f"bbox={row['_bbox']}"
            )

    # ========================================================
    # ONLY NEW
    # ========================================================

    if only_new:
        logging.info("")
        logging.info(
            "--- ONLY NEW DETECTIONS ---"
        )

        for row in only_new[
            :DIFF_LOG_LIMIT
        ]:
            logging.info(
                f"ONLY NEW | "
                f"freeze_id={row['freeze_id']} | "
                f"result={row['result']} | "
                f"person_id={row['person_id']} | "
                f"cluster={row['unknown_cluster_tag']} | "
                f"category_id={row['category_id']} | "
                f"bbox={row['_bbox']}"
            )

    # ========================================================
    # DIFFS
    # ========================================================

    if diffs:
        logging.info("")
        logging.info(
            "--- MATCHED FACE DIFFERENCES ---"
        )

        for diff in diffs[
            :DIFF_LOG_LIMIT
        ]:
            logging.info(
                "DIFF | "
                f"freeze_id="
                f"{diff['freeze_id']} | "
                f"IoU={diff['iou']:.6f} | "
                f"old_person="
                f"{diff['old_person_id']} | "
                f"new_person="
                f"{diff['new_person_id']} | "
                f"new_cluster="
                f"{diff['new_cluster']} | "
                + "; ".join(
                    diff["details"]
                )
            )

    # ========================================================
    # UNKNOWN SPLITS
    # ========================================================

    if split_old_clusters:
        logging.info("")
        logging.info(
            "--- UNKNOWN CLUSTER SPLITS ---"
        )

        for (
            old_person_id,
            new_clusters,
        ) in list(
            split_old_clusters.items()
        )[:DIFF_LOG_LIMIT]:

            logging.info(
                f"SPLIT | "
                f"old_person_id="
                f"{old_person_id} | "
                f"new_clusters="
                f"{sorted(new_clusters)}"
            )

    # ========================================================
    # UNKNOWN MERGES
    # ========================================================

    if merged_new_clusters:
        logging.info("")
        logging.info(
            "--- UNKNOWN CLUSTER MERGES ---"
        )

        for (
            new_cluster,
            old_person_ids,
        ) in list(
            merged_new_clusters.items()
        )[:DIFF_LOG_LIMIT]:

            logging.info(
                f"MERGE | "
                f"new_cluster="
                f"{new_cluster} | "
                f"old_person_ids="
                f"{sorted(old_person_ids)}"
            )


# ============================================================
# PROCESS ALL TEST MEDIA
# ============================================================

def process_test():
    db = SessionLocal()

    try:
        logging.info(
            "Loading InsightFace..."
        )

        face_model = get_insightface()

        logging.info(
            "Loading CLIP..."
        )

        (
            clip_model,
            clip_preprocess,
            clip_text_features,
            clip_prompt_categories,
        ) = get_clip()

        face_categories = (
            load_face_categories(db)
        )

        person_lookup = (
            load_person_lookup(db)
        )

        # ----------------------------------------------------
        # REFERENCE INDEX
        # ----------------------------------------------------

        logging.info(
            "Building REFERENCE FAISS..."
        )

        reference_index = (
            ReferenceIndex()
        )

        reference_index.build(db)

        # ----------------------------------------------------
        # UNKNOWN INDEX
        # ----------------------------------------------------

        logging.info(
            "Creating EMPTY UNKNOWN FAISS "
            "for clean regression test..."
        )

        unknown_index = (
            UnknownTestIndex(
                dimension=512
            )
        )

        # ----------------------------------------------------
        # MEDIA
        # ----------------------------------------------------

        medias = get_test_media(db)

        stats = defaultdict(int)

        all_results = []

        start_processing = (
            datetime.now()
        )

        total_media = len(medias)

        for media_index, media in enumerate(
            medias,
            start=1,
        ):
            freezes = (
                db.query(DBFreeze)
                .filter(
                    DBFreeze.media_id
                    == media.id
                )
                .order_by(
                    DBFreeze.time_in
                )
                .all()
            )

            stats["media"] += 1

            for freeze in freezes:
                rows = process_freeze(
                    db=db,

                    face_model=
                        face_model,

                    clip_model=
                        clip_model,

                    clip_preprocess=
                        clip_preprocess,

                    clip_text_features=
                        clip_text_features,

                    clip_prompt_categories=
                        clip_prompt_categories,

                    media=media,
                    freeze=freeze,

                    reference_index=
                        reference_index,

                    unknown_index=
                        unknown_index,

                    face_categories=
                        face_categories,

                    person_lookup=
                        person_lookup,

                    stats=stats,
                )

                all_results.extend(rows)

            if (
                media_index
                % PROGRESS_EVERY_MEDIA
                == 0
                or media_index
                == total_media
            ):
                elapsed = (
                    datetime.now()
                    - start_processing
                )

                logging.info(
                    f"Progress: "
                    f"{media_index}/"
                    f"{total_media} media | "
                    f"freezes="
                    f"{stats['freezes']} | "
                    f"buffalo_faces="
                    f"{stats['buffalo_faces']} | "
                    f"created="
                    f"{len(all_results)} | "
                    f"known="
                    f"{stats['known']} | "
                    f"unknown_index="
                    f"{unknown_index.index.ntotal} | "
                    f"elapsed={elapsed}"
                )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_results_csv(
            all_results,
            V5_RESULT_CSV,
        )

        # ----------------------------------------------------
        # PROCESSING SUMMARY
        # ----------------------------------------------------

        logging.info("")
        logging.info(
            "=" * 70
        )
        logging.info(
            "V5 TEST PROCESSING SUMMARY"
        )
        logging.info(
            "=" * 70
        )

        logging.info(
            f"Media:                  "
            f"{stats['media']}"
        )

        logging.info(
            f"Freezes processed:      "
            f"{stats['freezes']}"
        )

        logging.info(
            f"Freezes without faces:  "
            f"{stats['freezes_without_faces']}"
        )

        logging.info(
            f"Buffalo faces:          "
            f"{stats['buffalo_faces']}"
        )

        logging.info(
            f"V5 output faces:        "
            f"{len(all_results)}"
        )

        logging.info("")

        logging.info(
            f"Known:                  "
            f"{stats['known']}"
        )

        logging.info(
            f"Unknown NEW cluster:    "
            f"{stats['unknown_new']}"
        )

        logging.info(
            f"Unknown existing:       "
            f"{stats['unknown_existing']}"
        )

        logging.info(
            f"Low quality:            "
            f"{stats['low_quality']}"
        )

        logging.info(
            f"Service category:       "
            f"{stats['service_category']}"
        )

        logging.info(
            f"Skipped low det_score:  "
            f"{stats['skipped_low_det_score']}"
        )

        logging.info(
            f"Cannot read:            "
            f"{stats['cannot_read']}"
        )

        logging.info(
            f"Cannot open PIL:        "
            f"{stats['cannot_open_pil']}"
        )

        logging.info(
            f"Unknown FAISS vectors:  "
            f"{unknown_index.index.ntotal}"
        )

        logging.info(
            f"Unknown clusters:       "
            f"{unknown_index.next_cluster_id - 1}"
        )

        logging.info(
            "=" * 70
        )

        # ----------------------------------------------------
        # CONTROL CSV
        # ----------------------------------------------------

        old_rows = load_control_csv(
            CONTROL_CSV
        )

        compare_results(
            db=db,

            old_rows=old_rows,
            new_rows=all_results,

            person_lookup=
                person_lookup,

            face_categories=
                face_categories,
        )

    finally:
        db.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    start = datetime.now()

    logging.info(
        f"Start: {start}"
    )

    logging.info(
        f"TEST_SOURCE_IDS="
        f"{TEST_SOURCE_IDS}"
    )

    logging.info(
        f"CONTROL_CSV="
        f"{CONTROL_CSV}"
    )

    logging.info(
        f"V5_RESULT_CSV="
        f"{V5_RESULT_CSV}"
    )

    try:
        process_test()

    except Exception:
        logging.exception(
            "Critical error during V5 test"
        )

        raise

    finish = datetime.now()

    logging.info(
        f"Finished: {finish}"
    )

    logging.info(
        f"Running time: "
        f"{finish - start}"
    )