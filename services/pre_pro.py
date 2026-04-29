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
