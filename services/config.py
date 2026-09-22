from pathlib import Path


# еталони
PERSONS_FOLDER = Path(r"D:\data\Persons")
NEW_WIKI_PATH = Path(r"C:\faces\baza\parsing_wikipedia.json")

# медіа для тестування - відео + фото
TEST_MP4_LIGHT_FOLDER = Path(r"D:\faces\baza\test_mp4_light")
TEST_FOLDER = Path(r"D:\faces\baza\test")
TEST_FREEZE_FOLDER = Path(r"D:\faces\baza\test_freeze")
USER_UPLOAD_FOLDER = Path(r"Z:\FACES\DATA\user_uploads")

# проксі для новин і діджиталу
PROXY_INTVNEWS_CATALOG = Path(r"C:\faces\baza\proxy_catalog_intvnews")
PROXY_DIGITAL_CATALOG = Path(r"C:\faces\baza\proxy_catalog_digital")
PROXY_INTVNEWS_DUPLICATE_FILE = Path(r"C:\faces\baza\proxy_intvnews_duplicates.json")
PROXY_DIGITAL_DUPLICATE_FILE = Path(r"C:\faces\baza\proxy_digital_duplicates.json")

PROXY_NEWS_FOLDER = Path("Y:/")
PROXY_DIGITAL_FOLDER = Path(r"Y:\Digital")

# опис медіа
CSV_FOLDER = Path(r"D:\FirebirdData\backups\inTVNews_UA1_2026-06-12_08-47")

# статистика по каталогам
INTVNEWS_STATISTIC = Path(r"C:\faces\baza\intvnews_statistic.csv")

# архів новин і діджитал
HIRES_NEWS_FOLDER = Path(r"V:/")
HIRES_DIGITAL_FOLDER = Path(r"V:\Digital")

HIRES_INTVNEWS_CATALOG = Path(r"C:\faces\baza\hires_catalog_intvnews")
HIRES_DIGITAL_CATALOG = Path(r"C:\faces\baza\hires_catalog_digital")
HIRES_INTVNEWS_DUPLICATE_FILE = Path(r"C:\faces\baza\hires_intvnews_duplicates.json")
HIRES_DIGITAL_DUPLICATE_FILE = Path(r"C:\faces\baza\hires_digital_duplicates.json")

# фрізи
INTVNEWS_FREEZE_FOLDER = Path(r"D:\freezes\freeze_intvnews_arc")
TEMPORARY_FREEZES_FOLDER = Path(r"Z:\temporary_freeze")


# faiss
FAISS_FOLDER = Path(r"D:\data\Faiss")
UNKNOWN_FAISS_INDEX_PATH = FAISS_FOLDER / "unknown_faces.faiss"
UNKNOWN_FAISS_PERSON_IDS_PATH = FAISS_FOLDER / "unknown_person_ids.npy"

# реідентифікація
REVIEW_JSON_FILE = Path(r"C:\faces\baza\auto_reidentify_review.json")



# не використовується
GARBAGE_FOLDER_NAMES = {"from_64", "img", "input", "Logs", "NH", "part_restore", "Rozsliduvachi", "Temp", "trash"}
