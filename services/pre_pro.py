import os

from config import PERSONS_FOLDER

root_folder = PERSONS_FOLDER

for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.lower().endswith(".json"):
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Видалено: {file_path}")
            except Exception as e:
                print(f"Помилка: {file_path} -> {e}")


deleted = 0

for folder in sorted(root_folder.rglob("*"), reverse=True):
    if folder.is_dir():
        try:
            if not any(folder.iterdir()):
                folder.rmdir()
                print(f"Deleted empty folder: {folder}")
                deleted += 1

        except Exception as e:
            print(f"Error: {folder} -> {e}")

print("-" * 50)
print(f"Total deleted: {deleted}")
