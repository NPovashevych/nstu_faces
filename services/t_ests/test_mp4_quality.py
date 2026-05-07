import subprocess
import time
from pathlib import Path

import cv2
import pandas as pd
from scenedetect import open_video, SceneManager, ContentDetector

from services.config import MXF_FOLDER, FOR_TEST


REPORT_PATH = FOR_TEST / "benchmark_results.xlsx"

VARIANTS = [
    {
        "name": "A_slow_crf15",
        "preset": "slow",
        "crf": "15",
        "audio_bitrate": "320k",
        "vf": "yadif=0:-1:0,unsharp=3:3:0.4:3:3:0.0",
    },
    {
        "name": "B_fast_crf20",
        "preset": "fast",
        "crf": "20",
        "audio_bitrate": "128k",
        "vf": "yadif=0:-1:0",
    },
    {
        "name": "C_veryfast_crf22",
        "preset": "veryfast",
        "crf": "22",
        "audio_bitrate": "128k",
        "vf": "yadif=0:-1:0",
    },
]


def run_cmd(cmd):
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def get_video_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    result = run_cmd(cmd)

    if result.returncode != 0:
        return 0.0

    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def convert_mxf_to_mp4(mxf_path: Path, mp4_path: Path, variant: dict):
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(mxf_path),

        "-map", "0:v:0",
        "-map", "0:a:0?",

        "-vf", variant["vf"],

        "-c:v", "libx264",
        "-preset", variant["preset"],
        "-crf", variant["crf"],
        "-pix_fmt", "yuv420p",

        "-c:a", "aac",
        "-b:a", variant["audio_bitrate"],

        "-movflags", "+faststart",
        str(mp4_path),
    ]

    return run_cmd(cmd)


def get_scenes(mp4_path: Path):
    video = open_video(str(mp4_path))

    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=30.0, min_scene_len=15))

    sm.detect_scenes(video)
    scenes = sm.get_scene_list()

    if not scenes:
        duration = video.duration.get_seconds()
        return [(0.0, duration)]

    return [(s.get_seconds(), e.get_seconds()) for s, e in scenes]


def extract_freezes(mp4_path: Path, scenes, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    count = 0

    for start, end in scenes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        filename = output_dir / f"{int(start)}.jpg"
        cv2.imwrite(str(filename), frame)
        count += 1

    cap.release()
    return count


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return round(path.stat().st_size / 1024 / 1024, 2)


def main():
    FOR_TEST.mkdir(parents=True, exist_ok=True)

    mxf_files = sorted(Path(MXF_FOLDER).glob("*.mxf"))[:15]

    rows = []

    total_video_duration_sec = 0.0

    for variant in VARIANTS:
        print(f"\n===== VARIANT: {variant['name']} =====")

        variant_mp4_dir = FOR_TEST / variant["name"] / "mp4"
        variant_freeze_dir = FOR_TEST / variant["name"] / "freeze"

        for idx, mxf_path in enumerate(mxf_files, start=1):
            print(f"[{idx}/{len(mxf_files)}] {mxf_path.name}")

            mp4_path = variant_mp4_dir / f"{mxf_path.stem}.mp4"
            freeze_dir = variant_freeze_dir / mxf_path.stem

            original_duration = get_video_duration(mxf_path)

            if variant["name"] == VARIANTS[0]["name"]:
                total_video_duration_sec += original_duration

            # 1. Convert
            t0 = time.perf_counter()
            convert_result = convert_mxf_to_mp4(mxf_path, mp4_path, variant)
            convert_time = time.perf_counter() - t0

            if convert_result.returncode != 0:
                rows.append({
                    "variant": variant["name"],
                    "file": mxf_path.name,
                    "status": "convert_error",
                    "video_duration_sec": original_duration,
                    "convert_time_sec": round(convert_time, 2),
                    "scene_detect_time_sec": None,
                    "freeze_extract_time_sec": None,
                    "total_process_time_sec": round(convert_time, 2),
                    "scenes_count": None,
                    "freezes_count": None,
                    "mp4_size_mb": None,
                    "error": convert_result.stderr[:1000],
                })
                continue

            # 2. Scene detect
            t1 = time.perf_counter()
            scenes = get_scenes(mp4_path)
            scene_time = time.perf_counter() - t1

            # 3. Freeze extract
            t2 = time.perf_counter()
            freezes_count = extract_freezes(mp4_path, scenes, freeze_dir)
            freeze_time = time.perf_counter() - t2

            total_time = convert_time + scene_time + freeze_time

            rows.append({
                "variant": variant["name"],
                "file": mxf_path.name,
                "status": "ok",
                "video_duration_sec": round(original_duration, 2),
                "video_duration_min": round(original_duration / 60, 2),
                "convert_time_sec": round(convert_time, 2),
                "scene_detect_time_sec": round(scene_time, 2),
                "freeze_extract_time_sec": round(freeze_time, 2),
                "total_process_time_sec": round(total_time, 2),
                "scenes_count": len(scenes),
                "freezes_count": freezes_count,
                "mp4_size_mb": file_size_mb(mp4_path),
                "preset": variant["preset"],
                "crf": variant["crf"],
                "audio_bitrate": variant["audio_bitrate"],
                "vf": variant["vf"],
                "error": "",
            })

    df = pd.DataFrame(rows)

    summary = (
        df[df["status"] == "ok"]
        .groupby("variant")
        .agg(
            files_count=("file", "count"),
            total_video_duration_sec=("video_duration_sec", "sum"),
            total_video_duration_min=("video_duration_min", "sum"),
            total_convert_time_sec=("convert_time_sec", "sum"),
            total_scene_detect_time_sec=("scene_detect_time_sec", "sum"),
            total_freeze_extract_time_sec=("freeze_extract_time_sec", "sum"),
            total_process_time_sec=("total_process_time_sec", "sum"),
            avg_convert_time_sec=("convert_time_sec", "mean"),
            avg_scene_detect_time_sec=("scene_detect_time_sec", "mean"),
            avg_freeze_extract_time_sec=("freeze_extract_time_sec", "mean"),
            avg_total_process_time_sec=("total_process_time_sec", "mean"),
            total_scenes=("scenes_count", "sum"),
            total_freezes=("freezes_count", "sum"),
            total_mp4_size_mb=("mp4_size_mb", "sum"),
            avg_mp4_size_mb=("mp4_size_mb", "mean"),
        )
        .reset_index()
    )

    with pd.ExcelWriter(REPORT_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="details", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    print("\nDONE")
    print(f"Excel saved: {REPORT_PATH}")
    print(f"Total source video duration: {round(total_video_duration_sec / 60, 2)} min")


if __name__ == "__main__":
    main()
