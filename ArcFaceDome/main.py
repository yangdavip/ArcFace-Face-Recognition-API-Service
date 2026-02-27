import os
import json
import uuid
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import numpy as np

from database import (
    get_db, init_db, FaceLibrary, FaceMember, 
    FaceLibrarySchema, FaceMemberSchema, PaginatedResponse
)
from face_service import face_service

app = FastAPI(title="ArcFace Face Recognition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def root():
    return {"message": "ArcFace Face Recognition API", "version": "1.0.0"}


@app.post("/api/libraries", response_model=FaceLibrarySchema)
def create_library(name: str = Form(...), description: str = Form(None), db: Session = Depends(get_db)):
    existing = db.query(FaceLibrary).filter(FaceLibrary.name == name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Library name already exists")
    
    library = FaceLibrary(name=name, description=description)
    db.add(library)
    db.commit()
    db.refresh(library)
    return library


@app.get("/api/libraries", response_model=List[FaceLibrarySchema])
def list_libraries(db: Session = Depends(get_db)):
    return db.query(FaceLibrary).all()


@app.get("/api/libraries/{library_id}", response_model=FaceLibrarySchema)
def get_library(library_id: int, db: Session = Depends(get_db)):
    library = db.query(FaceLibrary).filter(FaceLibrary.id == library_id).first()
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    return library


@app.put("/api/libraries/{library_id}", response_model=FaceLibrarySchema)
def update_library(library_id: int, name: str = Form(None), description: str = Form(None), db: Session = Depends(get_db)):
    library = db.query(FaceLibrary).filter(FaceLibrary.id == library_id).first()
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    
    if name:
        existing = db.query(FaceLibrary).filter(FaceLibrary.name == name, FaceLibrary.id != library_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Library name already exists")
        library.name = name
    
    if description is not None:
        library.description = description
    
    db.commit()
    db.refresh(library)
    return library


@app.delete("/api/libraries/{library_id}")
def delete_library(library_id: int, db: Session = Depends(get_db)):
    library = db.query(FaceLibrary).filter(FaceLibrary.id == library_id).first()
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    
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
    
    items = [{"id": m.id, "name": m.name, "image_path": m.image_path, "created_at": m.created_at.isoformat()} for m in members]
    
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
    
    file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = UPLOAD_DIR / filename
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    try:
        embedding, face_info = face_service.extract_embedding(str(file_path))
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Face extraction failed: {str(e)}")
    
    embedding_str = json.dumps(embedding.tolist())
    
    member = FaceMember(
        library_id=library_id,
        name=name,
        embedding=float(np.linalg.norm(embedding)),
        embedding_vector=embedding_str,
        image_path=str(file_path)
    )
    db.add(member)
    db.commit()
    db.refresh(member)
    
    return {
        "id": member.id,
        "name": member.name,
        "image_path": member.image_path,
        "face_info": face_info,
        "created_at": member.created_at.isoformat()
    }


@app.post("/api/libraries/{library_id}/members/by-path")
def add_library_member_by_path(
    library_id: int,
    name: str = Form(...),
    image_path: str = Form(...),
    db: Session = Depends(get_db)
):
    library = db.query(FaceLibrary).filter(FaceLibrary.id == library_id).first()
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    
    if not Path(image_path).exists():
        raise HTTPException(status_code=400, detail="Image file not found")
    
    try:
        embedding, face_info = face_service.extract_embedding(image_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Face extraction failed: {str(e)}")
    
    embedding_str = json.dumps(embedding.tolist())
    
    member = FaceMember(
        library_id=library_id,
        name=name,
        embedding=float(np.linalg.norm(embedding)),
        embedding_vector=embedding_str,
        image_path=image_path
    )
    db.add(member)
    db.commit()
    db.refresh(member)
    
    return {
        "id": member.id,
        "name": member.name,
        "image_path": member.image_path,
        "face_info": face_info,
        "created_at": member.created_at.isoformat()
    }


@app.put("/api/libraries/{library_id}/members/{member_id}")
def update_library_member(
    library_id: int,
    member_id: int,
    name: str = Form(None),
    file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    member = db.query(FaceMember).filter(FaceMember.id == member_id, FaceMember.library_id == library_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    
    if name:
        member.name = name
    
    if file:
        if member.image_path:
            old_path = Path(member.image_path)
            old_path.unlink(missing_ok=True)
        
        file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        filename = f"{uuid.uuid4()}.{file_ext}"
        file_path = UPLOAD_DIR / filename
        
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        
        try:
            embedding, face_info = face_service.extract_embedding(str(file_path))
        except Exception as e:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Face extraction failed: {str(e)}")
        
        member.embedding = float(np.linalg.norm(embedding))
        member.embedding_vector = json.dumps(embedding.tolist())
        member.image_path = str(file_path)
    
    db.commit()
    db.refresh(member)
    
    return {
        "id": member.id,
        "name": member.name,
        "image_path": member.image_path,
        "updated_at": member.updated_at.isoformat()
    }


@app.delete("/api/libraries/{library_id}/members/{member_id}")
def delete_library_member(library_id: int, member_id: int, db: Session = Depends(get_db)):
    member = db.query(FaceMember).filter(FaceMember.id == member_id, FaceMember.library_id == library_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    
    if member.image_path:
        file_path = Path(member.image_path)
        file_path.unlink(missing_ok=True)
    
    db.delete(member)
    db.commit()
    
    return {"message": "Member deleted successfully"}


@app.post("/api/search")
def search_face(
    library_id: int = Form(...),
    file: UploadFile = File(...),
    top_k: int = Form(10),
    threshold: float = Form(0.5),
    db: Session = Depends(get_db)
):
    library = db.query(FaceLibrary).filter(FaceLibrary.id == library_id).first()
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    
    file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = UPLOAD_DIR / filename
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    try:
        query_embedding, face_info = face_service.extract_embedding(str(file_path))
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Face extraction failed: {str(e)}")
    
    file_path.unlink(missing_ok=True)
    
    members = db.query(FaceMember).filter(FaceMember.library_id == library_id).all()
    
    gallery = []
    for m in members:
        emb = np.array(json.loads(m.embedding_vector))
        gallery.append((m.id, emb, m.name))
    
    results = face_service.search_faces(query_embedding, gallery, top_k, threshold)
    
    return {
        "query_face": face_info,
        "results": results
    }


@app.post("/api/detect")
def detect_face(file: UploadFile = File(...)):
    file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = UPLOAD_DIR / filename
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    try:
        faces = face_service.detect_faces(str(file_path))
    finally:
        file_path.unlink(missing_ok=True)
    
    return {"faces": faces, "count": len(faces)}


@app.post("/api/detect/confidence")
def detect_face_with_confidence(file: UploadFile = File(...)):
    file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = UPLOAD_DIR / filename
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    try:
        faces = face_service.detect_faces_with_confidence(str(file_path))
    finally:
        file_path.unlink(missing_ok=True)
    
    return {"faces": faces, "count": len(faces)}
