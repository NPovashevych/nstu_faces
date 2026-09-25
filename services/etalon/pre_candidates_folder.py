import os

from services.config import CANDIDATE_FOLDER

from pathlib import Path
import shutil

ROOT = CANDIDATE_FOLDER


def change_files_stucture(root: Path):
    if not root.exists():
        print("ПОМИЛКА: ROOT не існує.")
        return

    if not root.is_dir():
        print("ПОМИЛКА: ROOT не є папкою.")
        return

    subroots = [p for p in root.iterdir() if p.is_dir()]
    print(f"Знайдено підкореневих папок: {len(subroots)}")

    moved_count = 0
    deleted_special_count = 0

    for subroot in subroots:
        print(f"Обробка: {subroot.name}")
        without_photo = subroot / "!without photo"
        if without_photo.exists() and without_photo.is_dir():
            print(f"  DELETE: {without_photo}")
            shutil.rmtree(without_photo)
            deleted_special_count += 1
        with_photo = subroot / "with PHOTO"
        if with_photo.exists() and with_photo.is_dir():
            print(f"  DELETE: {with_photo}")
            shutil.rmtree(with_photo)
            deleted_special_count += 1

        folders_to_move = [p for p in subroot.iterdir() if p.is_dir()]
        for source in folders_to_move:
            destination = root / source.name
            print(f"  MOVE: {source.name}")
            if destination.exists():
                print(f"    існуюча папка буде замінена: {destination}")
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(source), str(destination))
            moved_count += 1
        print(f"  DELETE SUBROOT: {subroot.name}")
        subroot.rmdir()
    print(f"Перенесено папок: {moved_count}")
    print(f"Видалено спеціальних папок: {deleted_special_count}")
    print(f"Оброблено підкореневих папок: {len(subroots)}")


def delete_json_file(root: Path):
    if not root.exists():
        print("ПОМИЛКА: ROOT не існує.")
        return

    if not root.is_dir():
        print("ПОМИЛКА: ROOT не є папкою.")
        return

    folders = [p for p in root.iterdir() if p.is_dir()]
    print(f"Знайдено папок: {len(folders)}")
    deleted_count = 0
    for folder in folders:
        json_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
        if json_files:
            print(f"{folder.name}: знайдено JSON: {len(json_files)}")
        for json_file in json_files:
            print(f"  DELETE: {json_file.name}")
            json_file.unlink()
            deleted_count += 1
    print(f"Оброблено папок: {len(folders)}")
    print(f"Видалено JSON-файлів: {deleted_count}")


if __name__ == "__main__":

    change_files_stucture(ROOT)
    delete_json_file(ROOT)
