from pathlib import Path

import yt_dlp


# url = "https://www.youtube.com/shorts/wxc6HK6frFQ"
url = "https://www.youtube.com/watch?v=pKbfvNpzCRg"


output_folder = Path(r"D:\faces\baza\youtube_test")
output_folder.mkdir(parents=True, exist_ok=True)

options = {
    "format": "bestvideo[ext=mp4][vcodec^=avc1][height<=720]/bestvideo[ext=mp4][height<=720]/bestvideo",
    "outtmpl": str(output_folder / "%(id)s.%(ext)s"),
    "noplaylist": True,
}

with yt_dlp.YoutubeDL(options) as ydl:
    info = ydl.extract_info(url, download=True)

    print("title:", info.get("title"))
    print("id:", info.get("id"))
    print("duration:", info.get("duration"))
    print("ext:", info.get("ext"))
