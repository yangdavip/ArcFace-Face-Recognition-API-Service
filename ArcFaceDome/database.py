import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional, List

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://yangda@localhost:5432/face_recognition"
)

engine = create_engine(DATABASE_URL)
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
    library_id = Column(Integer, ForeignKey("face_libraries.id"), nullable=False)
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
    library_id: int
    name: str
    image_path: Optional[str] = None
    
    class Config:
        from_attributes = True


class FaceMemberWithEmbedding(FaceMemberSchema):
    embedding: float
    embedding_vector: str


class PaginatedResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
