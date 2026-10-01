import re


def validate_ukrainian_text(value: str, field_name: str) -> str:
    value = " ".join(value.strip().split())

    if not value:
        raise ValueError(f"{field_name} is empty")

    if not re.fullmatch(r"[А-Яа-яІіЇїЄєҐґ'’ʼ\-\s]+", value):
        raise ValueError(f"{field_name} має бути українською")

    return value


def validate_ukrainian_pseudonym(value: str | None) -> str:
    value = " ".join((value or "").strip().split())

    if value and re.search(r"[A-Za-zЁёЫыЭэЪъ]", value):
        raise ValueError("Псевдонім має бути українською")

    return value


def validate_q_code(q_code: str | None) -> str:
    q_code = (q_code or "").strip().upper()

    if q_code and not re.fullmatch(r"Q\d{1,10}", q_code):
        raise ValueError("Q-код має бути у форматі Q + 1–10 цифр")

    return q_code
