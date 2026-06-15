import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional, List, Any

from config_loader import get_database_url

DATABASE_URL = os.getenv("DATABASE_URL") or get_database_url()

connect_args = {}
pool_kwargs = {}
if DATABASE_URL and DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
else:
    pool_kwargs = {
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,
    }

engine = create_engine(DATABASE_URL, connect_args=connect_args, **pool_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class FaceLibrary(Base):
    __tablename__ = "face_libraries"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class FaceMember(Base):
    __tablename__ = "face_members"
    
    id = Column(Integer, primary_key=True, index=True)
    record_id = Column(String(36), unique=True, nullable=False, index=True)
    library_id = Column(Integer, ForeignKey("face_libraries.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    embedding = Column(Float, nullable=False)
    embedding_vector = Column(Text, nullable=False)
    image_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class FaceLibrarySchema(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    
    class Config:
        from_attributes = True


class FaceMemberSchema(BaseModel):
    id: Optional[int] = None
    record_id: str
    library_id: int
    name: str
    image_path: Optional[str] = None
    created_at: Optional[str] = None
    
    class Config:
        from_attributes = True


class FaceMemberWithEmbedding(FaceMemberSchema):
    embedding: float
    embedding_vector: str


class PaginatedResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[Any]


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
