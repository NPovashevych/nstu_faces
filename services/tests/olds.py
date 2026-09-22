def check_video_quality(video_path: Path) -> bool:
    info = ffprobe_info(video_path)

    if info is None:
        return False

    if not has_video_stream(info):
        logging.warning(f"No video stream: {video_path}")
        return False

    if not has_audio_stream(info):
        logging.warning(f"No audio stream: {video_path}")

    duration = float(info.get("format", {}).get("duration", 0) or 0)

    if duration <= 0:
        logging.warning(f"Bad duration: {video_path}")
        return False

    return check_playable(video_path)