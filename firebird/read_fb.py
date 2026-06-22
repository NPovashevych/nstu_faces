import csv
import fdb
from pathlib import Path
from datetime import date


month = 1
year = 2020

DATE_FROM = date(year, month, 1)
DATE_TO = date(year, month + 1, 1)


DB_PATH = r"C:\faces\FirebirdData\restored\ARCHIVE.FDB"
OUT_PATH = Path(fr"C:\faces\FirebirdData\backups\inTVNews_UA1_2026-06-12_08-47\materials_{DATE_FROM}-{DATE_TO}.csv")

con = fdb.connect(
    dsn=DB_PATH,
    user="FBADMIN",
    password="fbadmin",
    charset="WIN1251",
)

def load_dict(table, id_col, name_col):
    cur = con.cursor()
    cur.execute(f"SELECT {id_col}, {name_col} FROM {table}")
    return {str(row[0]): row[1] for row in cur.fetchall()}

authors_dict = load_dict("AUTHORS", "AUTHOR_ID", "AUTHOR_NAME")
operators_dict = load_dict("OPERATORS", "OPERATOR_ID", "OPERATOR_NAME")
section_dict = load_dict("SECTIONS", "SECTION_ID", "SECTION_NAME")
# channel_dict = load_dict("CHANNELS_ID_LIST", "CHANNEL_ID", "CHANNEL_NAME")


def names_from_list(value, dictionary):
    if not value:
        return ""
    ids = [x.strip() for x in str(value).split(",") if x.strip()]
    names = []
    for x in ids:
        name = dictionary.get(x)
        if name:
            names.append(str(name))
        else:
            names.append(x)
    return ", ".join(names)

cur = con.cursor()
cur.execute("""
SELECT MATERIAL_ID, SECTION_ID, SHOOTING_DATE_FROM, AUTHORS_ID_LIST, OPERATORS_ID_LIST, MATERIAL_DESCRIPTION
FROM MATERIALS
WHERE SHOOTING_DATE_FROM >= ? AND SHOOTING_DATE_FROM < ?
ORDER BY SHOOTING_DATE_FROM, MATERIAL_ID """,
(DATE_FROM, DATE_TO))

with OUT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["material_id", "section", "shooting_date", "journalists", "operators", "description"])
    for row in cur:
        (material_id, section_id, shooting_date, authors_ids, operators_ids, description) = row

        writer.writerow([
            material_id,
            names_from_list(section_id, section_dict),
            shooting_date,
            names_from_list(authors_ids, authors_dict),
            names_from_list(operators_ids, operators_dict),
            description,
        ])

con.close()

print(f"OK: saved to {OUT_PATH}")
