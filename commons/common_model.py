from insightface.app import FaceAnalysis
import logging


FACE_DET_SIZE = 640


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
