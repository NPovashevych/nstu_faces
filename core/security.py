from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    if len(password.encode("utf-8")) > 72:
        raise ValueError("Password is too long for bcrypt. Max 72 bytes.")

    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if len(plain_password.encode("utf-8")) > 72:
        return False

    return pwd_context.verify(plain_password, hashed_password)
