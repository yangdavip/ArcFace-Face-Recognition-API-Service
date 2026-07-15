import os
import json
import uuid
import logging
import time
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, delete
from sqlalchemy import text as _sa_text
from pydantic import BaseModel, Field
import numpy as np

from sqlalchemy.orm import Session
from sqlalchemy import select, text, delete
from sqlalchemy import text as _sa_text
from pydantic import BaseModel, Field
import numpy as np

from database import (
    get_db, init_db, FaceLibrary, FaceMember, 
    FaceLibrarySchema, FaceMemberSchema, PaginatedResponse
)
from face_service import face_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Model warmup
    import numpy as np
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        await face_service.detect_faces_async(dummy)
    except Exception:
        pass
    yield
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    try:
        face_service.detect_faces(dummy_image)
        logger.info("Model warmup completed")
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")
    yield


app = FastAPI(title="ArcFace Face Recognition API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("API_KEY", "")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if API_KEY and request.url.path.startswith("/api/"):
        api_key = request.headers.get("X-API-Key")
        if api_key != API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key. Provide via X-API-Key header."})
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    
    body = None
    if request.method in ["POST", "PUT", "PATCH"] and not request.url.path.startswith("/api/detect"):
        try:
            body = await request.body()
            if body:
                try:
                    body = json.loads(body)
                    if isinstance(body, dict):
                        if 'image' in body:
                            body = {**body, 'image': body['image'][:30] + '...(base64 truncated)'}
                        if 'file' in body:
                            body = {**body, 'file': body['file'][:30] + '...(base64 truncated)'}
                except:
                    body = str(body)[:200]
        except:
            pass
    
    logger.info(f"[{req_id}] 📥 {request.method} {request.url.path}")
    if body:
        logger.info(f"[{req_id}] 📥 Body: {body}")
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    
    process_time = time.time() - start_time
    logger.info(f"[{req_id}] 📤 {response.status_code} | {process_time:.3f}s")
    
    return response

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def is_path_in_upload_dir(path: Path) -> bool:
    try:
        return path.resolve().absolute().parts[:len(UPLOAD_DIR.parts)] == UPLOAD_DIR.parts
    except (ValueError, OSError):
        return False


import base64
import io
from PIL import Image
from config_loader import get_upload_config

_upload_config = get_upload_config()
ALLOWED_EXTENSIONS = set(_upload_config.get("allowed_extensions", ["jpg", "jpeg", "png", "bmp"]))
MAX_FILE_SIZE = _upload_config.get("max_file_size", 10 * 1024 * 1024)

_MAGIC_BYTES = {
    b'\xff\xd8\xff': ('jpg', 'jpeg'),
    b'\x89PNG\r\n\x1a\n': ('png',),
    b'BM': ('bmp',),
}


def _get_file_ext(filename: str) -> str:
    return filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''


def validate_upload(filename: str, file_size: int, content: bytes):
    ext = _get_file_ext(filename)
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type '.{ext}' not allowed")

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File exceeds max size of {MAX_FILE_SIZE} bytes")

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    matched = False
    for magic, exts in _MAGIC_BYTES.items():
        if content[:len(magic)] == magic:
            if ext and ext not in exts:
                raise HTTPException(status_code=400, detail="File extension does not match content type")
            matched = True
            break
    if not matched:
        raise HTTPException(status_code=400, detail="File is not a supported image format")


def decode_base64_image(base64_str: str) -> bytes:
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    # 修复缺失的 base64 padding
    padding = 4 - len(base64_str) % 4
    if padding != 4:
        base64_str += '=' * padding
    image_data = base64.b64decode(base64_str)
    validate_upload("image.jpg", len(image_data), image_data)
    return image_data


def save_image_bytes(image_bytes: bytes) -> Path:
    """Save image bytes to upload directory and return path."""
    filename = f"{uuid.uuid4()}.jpg"
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    return file_path


class Base64Request(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    image: str


class Base64SearchRequest(BaseModel):
    image: str
    top_k: int = Field(default=10, ge=1, le=1000)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class Base64DetectRequest(BaseModel):
    image: str


class UpdateLibraryRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = None


class AddMemberRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class SearchRequest(BaseModel):
    top_k: int = Field(default=10, ge=1, le=1000)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

@app.get("/")
async def root():
    return {"message": "ArcFace Face Recognition API", "version": "1.0.0"}


@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "database": str(e)})


class CreateLibraryRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None


@app.post("/api/libraries", response_model=FaceLibrarySchema)
def create_library(request: CreateLibraryRequest, db: Session = Depends(get_db)):
    existing = db.query(FaceLibrary).filter(FaceLibrary.name == request.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Library name already exists")
    
    library = FaceLibrary(name=request.name, description=request.description)
    db.add(library)
    db.commit()
    db.refresh(library)
    return library


@app.get("/api/libraries", response_model=List[FaceLibrarySchema])
def list_libraries(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    query = db.query(FaceLibrary).order_by(FaceLibrary.id).offset((page - 1) * page_size).limit(page_size)
    return query.all()


@app.get("/api/libraries/{library_id}", response_model=FaceLibrarySchema)
def get_library(library_id: int, db: Session = Depends(get_db)):
    library = db.query(FaceLibrary).filter(FaceLibrary.id == library_id).first()
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    return library


@app.put("/api/libraries/{library_id}", response_model=FaceLibrarySchema)
def update_library(library_id: int, request: UpdateLibraryRequest, db: Session = Depends(get_db)):
    library = db.query(FaceLibrary).filter(FaceLibrary.id == library_id).first()
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    
    if request.name is not None:
        existing = db.query(FaceLibrary).filter(FaceLibrary.name == request.name, FaceLibrary.id != library_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Library name already exists")
        library.name = request.name
    
    if request.description is not None:
        library.description = request.description
    
    db.commit()
    db.refresh(library)
    return library


@app.delete("/api/libraries/{library_id}")
def delete_library(library_id: int, db: Session = Depends(get_db)):
    library = db.query(FaceLibrary).filter(FaceLibrary.id == library_id).first()
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    
    members = db.query(FaceMember).filter(FaceMember.library_id == library_id).all()
    for m in members:
        if m.image_path and is_path_in_upload_dir(Path(m.image_path)):
            Path(m.image_path).unlink(missing_ok=True)
    db.query(FaceMember).filter(FaceMember.library_id == library_id).delete()
    db.delete(library)
    db.commit()
    return {"message": "Library deleted successfully"}


@app.get("/api/libraries/{library_id}/members", response_model=PaginatedResponse)
def list_library_members(
    library_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    library = db.query(FaceLibrary).filter(FaceLibrary.id == library_id).first()
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    
    total = db.query(FaceMember).filter(FaceMember.library_id == library_id).count()
    members = db.query(FaceMember).filter(FaceMember.library_id == library_id).offset((page - 1) * page_size).limit(page_size).all()
    
    items = [{"id": m.id, "record_id": m.record_id, "name": m.name, "image_path": m.image_path, "created_at": m.created_at.isoformat(), "updated_at": m.updated_at.isoformat() if m.updated_at else None} for m in members]
    
    return {"total": total, "page": page, "page_size": page_size, "items": items}


@app.post("/api/libraries/{library_id}/members")
def add_library_member(
    library_id: int,
    name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    library = db.query(FaceLibrary).filter(FaceLibrary.id == library_id).first()
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    
    file_bytes = file.file.read()
    validate_upload(file.filename or "image.jpg", len(file_bytes), file_bytes)
    file_ext = _get_file_ext(file.filename or "image.jpg") or 'jpg'
    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = UPLOAD_DIR / filename
    
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    
    try:
        embedding, face_info = face_service.extract_embedding(file_bytes)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Face extraction failed: {str(e)}")
    
    embedding_bytes = embedding.astype(np.float32).tobytes()
    
    member = FaceMember(
        record_id=str(uuid.uuid4()),
        library_id=library_id,
        name=name,
        embedding_vector=embedding_bytes,
        image_path=str(file_path)
    )
    db.add(member)
    try:
        db.commit()
    except Exception:
        file_path.unlink(missing_ok=True)
        db.rollback()
        raise
    db.refresh(member)
    
    return {
        "id": member.id,
        "record_id": member.record_id,
        "name": member.name,
        "image_path": member.image_path,
        "face_info": face_info,
        "created_at": member.created_at.isoformat()
    }


class AddMemberByPathRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    image_path: str


@app.post("/api/libraries/{library_id}/members/by-path")
def add_library_member_by_path(
    library_id: int,
    request: AddMemberByPathRequest,
    db: Session = Depends(get_db)
):
    library = db.query(FaceLibrary).filter(FaceLibrary.id == library_id).first()
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    
    src = Path(request.image_path).resolve()
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=400, detail="Image file not found")
    
    file_ext = src.suffix[1:] if src.suffix else 'jpg'
    filename = f"{uuid.uuid4()}.{file_ext}"
    dest = UPLOAD_DIR / filename
    try:
        import shutil
        shutil.copy2(src, dest)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to copy image: {str(e)}")
    
    file_bytes = dest.read_bytes()
    validate_upload(dest.name, len(file_bytes), file_bytes)
    
    try:
        embedding, face_info = face_service.extract_embedding(file_bytes)
    except Exception:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Face extraction failed")
    
    embedding_bytes = embedding.astype(np.float32).tobytes()
    
    member = FaceMember(
        record_id=str(uuid.uuid4()),
        library_id=library_id,
        name=request.name,
        embedding_vector=embedding_bytes,
        image_path=str(dest)
    )
    db.add(member)
    try:
        db.commit()
    except Exception:
        dest.unlink(missing_ok=True)
        db.rollback()
        raise
    db.refresh(member)
    
    return {
        "id": member.id,
        "record_id": member.record_id,
        "name": member.name,
        "image_path": member.image_path,
        "face_info": face_info,
        "created_at": member.created_at.isoformat()
    }


class UpdateMemberRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    image: str | None = None


@app.put("/api/libraries/{library_id}/members/{member_id}")
def update_library_member(
    library_id: int,
    member_id: int,
    request: UpdateMemberRequest,
    db: Session = Depends(get_db)
):
    member = db.query(FaceMember).filter(FaceMember.id == member_id, FaceMember.library_id == library_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    
    if request.name is not None:
        member.name = request.name
    
    if request.image:
        try:
            file_path = decode_base64_image(request.image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
        
        if member.image_path and is_path_in_upload_dir(Path(member.image_path)):
            Path(member.image_path).unlink(missing_ok=True)
        
        try:
            embedding, face_info = face_service.extract_embedding(str(file_path))
        except Exception as e:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Face extraction failed: {str(e)}")
        
        member.embedding_vector = embedding.astype(np.float32).tobytes()
        member.image_path = str(file_path)
    
    db.commit()
    db.refresh(member)
    
    return {
        "id": member.id,
        "record_id": member.record_id,
        "name": member.name,
        "image_path": member.image_path,
        "updated_at": member.updated_at.isoformat()
    }


@app.get("/api/libraries/{library_id}/members/by-record/{record_id}")
def get_member_by_record_id(library_id: int, record_id: str, db: Session = Depends(get_db)):
    member = db.query(FaceMember).filter(FaceMember.record_id == record_id, FaceMember.library_id == library_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    
    return {
        "id": member.id,
        "record_id": member.record_id,
        "name": member.name,
        "image_path": member.image_path,
        "created_at": member.created_at.isoformat()
    }


@app.delete("/api/libraries/{library_id}/members/by-record/{record_id}")
def delete_member_by_record_id(library_id: int, record_id: str, db: Session = Depends(get_db)):
    member = db.query(FaceMember).filter(FaceMember.record_id == record_id, FaceMember.library_id == library_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    
    if member.image_path and is_path_in_upload_dir(Path(member.image_path)):
        Path(member.image_path).unlink(missing_ok=True)
    
    db.delete(member)
    db.commit()
    
    return {"message": "Member deleted successfully"}


@app.delete("/api/libraries/{library_id}/members/{member_id}")
def delete_library_member(library_id: int, member_id: int, db: Session = Depends(get_db)):
    member = db.query(FaceMember).filter(FaceMember.id == member_id, FaceMember.library_id == library_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    
    if member.image_path and is_path_in_upload_dir(Path(member.image_path)):
        Path(member.image_path).unlink(missing_ok=True)
    
    db.delete(member)
    db.commit()
    
    return {"message": "Member deleted successfully"}


class SearchJsonRequest(BaseModel):
    library_id: int | None = None
    image: str | None = None
    file: str | None = None
    top_k: int = Field(default=10, ge=1, le=1000)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


@app.post("/api/search")
def search_face(
    library_id: Optional[int] = Form(None),
    file: UploadFile = File(None),
    top_k: int = Form(10),
    threshold: float = Form(0.5),
    image: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    if not library_id:
        raise HTTPException(status_code=400, detail="library_id is required")
    
    library = db.query(FaceLibrary).filter(FaceLibrary.id == library_id).first()
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    
    if file:
        file_bytes = file.file.read()
        validate_upload(file.filename or "image.jpg", len(file_bytes), file_bytes)
        # No temp file needed - process directly from bytes
        try:
            query_embedding, face_info = face_service.extract_embedding(file_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Face extraction failed: {str(e)}")
    elif image:
        try:
            image_bytes = decode_base64_image(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
        
        try:
            query_embedding, face_info = face_service.extract_embedding(image_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Face extraction failed: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="file or image is required")
    
    members = db.query(FaceMember).filter(FaceMember.library_id == library_id).all()
    
    member_ids = [m.id for m in members]
    names = [m.name for m in members]
    embeddings_matrix = np.frombuffer(b"".join(m.embedding_vector for m in members), dtype=np.float32).reshape(-1, 512)
    
    results = face_service.search_faces(query_embedding, embeddings_matrix, member_ids, names, top_k, threshold)
    
    return {
        "query_face": face_info,
        "results": results
    }


@app.post("/api/search/json")
def search_face_json(
    request: SearchJsonRequest,
    db: Session = Depends(get_db)
):
    if request.library_id is None:
        raise HTTPException(status_code=400, detail="library_id is required")
    library = db.query(FaceLibrary).filter(FaceLibrary.id == request.library_id).first()
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    
    base64_image = request.image or request.file
    if not base64_image:
        raise HTTPException(status_code=400, detail="image or file is required")
    
    try:
        image_bytes = decode_base64_image(base64_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
    
    try:
        query_embedding, face_info = face_service.extract_embedding(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Face extraction failed: {str(e)}")
    
    members = db.query(FaceMember).filter(FaceMember.library_id == request.library_id).all()
    
    member_ids = [m.id for m in members]
    names = [m.name for m in members]
    embeddings_matrix = np.frombuffer(b"".join(m.embedding_vector for m in members), dtype=np.float32).reshape(-1, 512)
    
    results = face_service.search_faces(query_embedding, embeddings_matrix, member_ids, names, request.top_k, request.threshold)
    
    return {
        "query_face": face_info,
        "results": results
    }


@app.post("/api/detect")
def detect_face(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    validate_upload(file.filename or "image.jpg", len(file_bytes), file_bytes)
    
    try:
        faces = face_service.detect_faces(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {"faces": faces, "count": len(faces)}


@app.post("/api/detect/confidence")
def detect_face_with_confidence(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    validate_upload(file.filename or "image.jpg", len(file_bytes), file_bytes)
    
    try:
        faces = face_service.detect_faces_with_confidence(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {"faces": faces, "count": len(faces)}


@app.post("/api/libraries/{library_id}/members/base64")
def add_member_by_base64(
    library_id: int,
    request: Base64Request,
    db: Session = Depends(get_db)
):
    library = db.query(FaceLibrary).filter(FaceLibrary.id == library_id).first()
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    
    try:
        image_bytes = decode_base64_image(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
    
    try:
        embedding, face_info = face_service.extract_embedding(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Face extraction failed: {str(e)}")
    
    # Save image to disk for reference (still needed for member records)
    filename = f"{uuid.uuid4()}.jpg"
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    
    embedding_bytes = embedding.astype(np.float32).tobytes()
    
    member = FaceMember(
        record_id=str(uuid.uuid4()),
        library_id=library_id,
        name=request.name,
        embedding_vector=embedding_bytes,
        image_path=str(file_path)
    )
    db.add(member)
    try:
        db.commit()
    except Exception:
        file_path.unlink(missing_ok=True)
        db.rollback()
        raise
    db.refresh(member)
    
    return {
        "id": member.id,
        "record_id": member.record_id,
        "name": member.name,
        "image_path": member.image_path,
        "face_info": face_info,
        "created_at": member.created_at.isoformat()
    }


@app.post("/api/search/base64")
def search_face_by_base64(
    library_id: int,
    request: Base64SearchRequest,
    db: Session = Depends(get_db)
):
    library = db.query(FaceLibrary).filter(FaceLibrary.id == library_id).first()
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    
    try:
        image_bytes = decode_base64_image(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
    
    try:
        query_embedding, face_info = face_service.extract_embedding(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Face extraction failed: {str(e)}")
    
    members = db.query(FaceMember).filter(FaceMember.library_id == library_id).all()
    
    member_ids = [m.id for m in members]
    names = [m.name for m in members]
    embeddings_matrix = np.frombuffer(b"".join(m.embedding_vector for m in members), dtype=np.float32).reshape(-1, 512)
    
    results = face_service.search_faces(query_embedding, embeddings_matrix, member_ids, names, request.top_k, request.threshold)
    
    return {
        "query_face": face_info,
        "results": results
    }


@app.post("/api/detect/base64")
def detect_face_by_base64(request: Base64DetectRequest):
    try:
        image_bytes = decode_base64_image(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
    
    try:
        faces = face_service.detect_faces(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Face detection failed: {str(e)}")
    
    return {"faces": faces, "count": len(faces)}


@app.post("/api/detect/confidence/base64")
def detect_face_confidence_by_base64(request: Base64DetectRequest):
    try:
        image_bytes = decode_base64_image(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
    
    try:
        faces = face_service.detect_faces_with_confidence(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Face detection failed: {str(e)}")
    
    return {"faces": faces, "count": len(faces)}


@app.post("/api/compare")
def compare_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    try:
        bytes1 = image1.file.read()
        bytes2 = image2.file.read()
        validate_upload(image1.filename or "image.jpg", len(bytes1), bytes1)
        validate_upload(image2.filename or "image.jpg", len(bytes2), bytes2)

        result = face_service.compare_faces(bytes1, bytes2)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    from database import DATABASE_URL

    import uvicorn
    workers = 1
    if DATABASE_URL and not DATABASE_URL.startswith("sqlite"):
        workers = 4
    else:
        logger.warning("SQLite detected — forcing workers=1 to avoid 'database is locked' errors. Use PostgreSQL for multi-worker deployment.")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=workers, reload=True)
