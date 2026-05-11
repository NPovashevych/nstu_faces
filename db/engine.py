from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from env import DATABASE_URL


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
