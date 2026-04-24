from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = "postgresql://faces_admin:12345@localhost:5432/faces_db"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

BASE = declarative_base()
