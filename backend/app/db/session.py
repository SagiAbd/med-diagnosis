from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

engine = create_engine(
    settings.get_database_url,
    pool_pre_ping=True,      # test connection before use, auto-reconnect if dropped
    pool_recycle=1800,       # recycle connections every 30 min
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 