from pathlib import Path
from datetime import datetime


INTVNEWS_FREEZE_FOLDER = Path(r"D:\freezes\freeze_intvnews_arc")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


def format_size(size_bytes: int) -> str:
    gb = size_bytes / (1024 ** 3)
    tb = size_bytes / (1024 ** 4)

    if tb >= 1:
        return f"{tb:.2f} TB"

    return f"{gb:.2f} GB"


def count_freezes(base_folder: Path):
    folder_count = 0
    freeze_count = 0
    total_size = 0

    start = datetime.now()

    print(f"Старт: {start}")
    print(f"Папка: {base_folder}")
    print()

    for folder in base_folder.iterdir():
        if not folder.is_dir():
            continue

        folder_count += 1

        for file in folder.iterdir():
            if not file.is_file():
                continue

            if file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            freeze_count += 1

            try:
                total_size += file.stat().st_size
            except OSError:
                pass

        if folder_count % 10_000 == 0:
            print(
                f"Папок: {folder_count:,} | "
                f"Фрізів: {freeze_count:,} | "
                f"Розмір: {format_size(total_size)}"
            )

    finish = datetime.now()

    print()
    print("---------------------------------------------")
    print(f"Папок:               {folder_count:,}")
    print(f"Фрізів:              {freeze_count:,}")
    print(f"Загальний розмір:    {format_size(total_size)}")

    if folder_count:
        print(
            f"Середньо фрізів/медіа: "
            f"{freeze_count / folder_count:.2f}"
        )

    print(f"Час підрахунку:       {finish - start}")
    print("---------------------------------------------")


if __name__ == "__main__":
    count_freezes(INTVNEWS_FREEZE_FOLDER)
