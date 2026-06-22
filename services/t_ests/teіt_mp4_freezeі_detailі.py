from pathlib import Path

import cv2
import pandas as pd
from insightface.app import FaceAnalysis

from old_versions.face_quality import is_good_face
from services.config import FOR_TEST


REPORT_PATH = FOR_TEST / "benchmark_results.xlsx"
MIN_FACE_QUALITY = 0.15


def load_model():
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def count_faces_in_freeze_folder(
    model,
    freeze_dir: Path,
    variant: str,
    file_name: str,
):
    total_faces = 0
    good_faces = 0
    low_quality_faces = 0
    frames_with_faces = 0
    frames_checked = 0

    freeze_details = []

    for img_path in sorted(freeze_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))

        if img is None:
            continue

        frames_checked += 1

        faces = model.get(img)

        freeze_total = len(faces)
        freeze_good = 0
        freeze_low = 0

        if faces:
            frames_with_faces += 1

        for face in faces:
            quality = is_good_face(img, face)

            total_faces += 1

            if quality < MIN_FACE_QUALITY:
                low_quality_faces += 1
                freeze_low += 1
            else:
                good_faces += 1
                freeze_good += 1

        freeze_details.append({
            "variant": variant,
            "file": file_name,
            "freeze_name": img_path.name,
            "faces_total": freeze_total,
            "faces_good_quality": freeze_good,
            "faces_low_quality": freeze_low,
        })

    summary = {
        "frames_checked": frames_checked,
        "frames_with_faces": frames_with_faces,
        "faces_total": total_faces,
        "faces_good_quality": good_faces,
        "faces_low_quality": low_quality_faces,
    }

    return summary, freeze_details


def main():
    df = pd.read_excel(REPORT_PATH, sheet_name="details")

    model = load_model()

    new_rows = []
    freeze_rows = []

    for idx, row in df.iterrows():
        if row["status"] != "ok":
            new_rows.append(row.to_dict())
            continue

        variant = row["variant"]
        file_name = row["file"]
        stem = Path(file_name).stem

        freeze_dir = FOR_TEST / variant / "freeze" / stem

        print(f"[{idx + 1}/{len(df)}] {variant} / {file_name}")

        if not freeze_dir.exists():
            result = {
                "frames_checked": 0,
                "frames_with_faces": 0,
                "faces_total": 0,
                "faces_good_quality": 0,
                "faces_low_quality": 0,
                "face_detect_error": f"Freeze dir not found: {freeze_dir}",
            }

            freeze_rows.append({
                "variant": variant,
                "file": file_name,
                "freeze_name": "",
                "faces_total": 0,
                "faces_good_quality": 0,
                "faces_low_quality": 0,
                "error": f"Freeze dir not found: {freeze_dir}",
            })

        else:
            result, details = count_faces_in_freeze_folder(
                model=model,
                freeze_dir=freeze_dir,
                variant=variant,
                file_name=file_name,
            )

            result["face_detect_error"] = ""
            freeze_rows.extend(details)

        row_dict = row.to_dict()
        row_dict.update(result)
        new_rows.append(row_dict)

    df_new = pd.DataFrame(new_rows)
    df_freezes = pd.DataFrame(freeze_rows)

    summary = (
        df_new[df_new["status"] == "ok"]
        .groupby("variant")
        .agg(
            files_count=("file", "count"),
            total_video_duration_sec=("video_duration_sec", "sum"),
            total_process_time_sec=("total_process_time_sec", "sum"),
            total_scenes=("scenes_count", "sum"),
            total_freezes=("freezes_count", "sum"),
            total_faces=("faces_total", "sum"),
            total_good_faces=("faces_good_quality", "sum"),
            total_low_quality_faces=("faces_low_quality", "sum"),
            frames_with_faces=("frames_with_faces", "sum"),
            total_mp4_size_mb=("mp4_size_mb", "sum"),
        )
        .reset_index()
    )

    with pd.ExcelWriter(REPORT_PATH, engine="openpyxl", mode="w") as writer:
        df_new.to_excel(writer, sheet_name="details", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        df_freezes.to_excel(writer, sheet_name="freeze_details", index=False)

    print("DONE")
    print(f"Updated Excel: {REPORT_PATH}")


if __name__ == "__main__":
    main()
