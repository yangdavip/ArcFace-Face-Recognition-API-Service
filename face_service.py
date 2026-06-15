import cv2
import numpy as np
from typing import List, Dict, Tuple
from insightface.app import FaceAnalysis
import onnxruntime
from config_loader import get_threshold_config


def get_providers():
    available = onnxruntime.get_available_providers()
    if 'CUDAExecutionProvider' in available:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def get_device_id():
    available = onnxruntime.get_available_providers()
    return 0 if 'CUDAExecutionProvider' in available else -1


class FaceService:
    def __init__(self, model_name='buffalo_l', providers=None, ctx_id=None):
        if providers is None:
            providers = get_providers()
        if ctx_id is None:
            ctx_id = get_device_id()
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("Failed to read image")
        
        faces = self.app.get(img)
        results = []
        
        for face in faces:
            results.append({
                'bbox': face.bbox.tolist(),
                'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
                'score': float(face.det_score) if hasattr(face, 'det_score') else 1.0,
            })
        
        return results
    
    def detect_faces_with_confidence(self, image_path: str) -> List[Dict]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("Failed to read image")
        
        faces = self.app.get(img)
        results = []
        
        for face in faces:
            face_info = {
                'bbox': face.bbox.tolist(),
                'det_score': float(face.det_score) if hasattr(face, 'det_score') else 1.0,
            }
            
            if hasattr(face, 'kps') and face.kps is not None:
                face_info['landmarks'] = face.kps.tolist()
            else:
                face_info['landmarks'] = None
            
            results.append(face_info)
        
        return results
    
    def extract_embedding(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("Failed to read image")
        
        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("No face detected in image")
        if len(faces) > 1:
            raise ValueError("Multiple faces detected in image")
        
        face = faces[0]
        embedding = face.embedding
        
        return embedding, {
            'bbox': face.bbox.tolist(),
            'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
            'det_score': float(face.det_score) if hasattr(face, 'det_score') else 1.0,
        }
    
    def search_faces(self, query_embedding: np.ndarray, embeddings_matrix: np.ndarray, member_ids: List, names: List[str], top_k: int = 10, threshold: float = None) -> List[Dict]:
        if threshold is None:
            threshold = get_threshold_config().get("cosine_similarity", 0.5)
        
        if embeddings_matrix.shape[0] == 0:
            return []
        
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        emb_norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        emb_norms[emb_norms == 0] = 1
        normalized_embs = embeddings_matrix / emb_norms
        cosine_sims = np.dot(normalized_embs, query_norm)
        
        mask = cosine_sims >= threshold
        if not np.any(mask):
            return []
        
        valid_indices = np.where(mask)[0]
        valid_sims = cosine_sims[valid_indices]
        sorted_order = np.argsort(-valid_sims)[:top_k]
        
        return [
            {
                'member_id': member_ids[valid_indices[i]],
                'name': names[valid_indices[i]],
                'similarity': float(valid_sims[i]),
                'similarity_percent': float((valid_sims[i] + 1) / 2 * 100),
            }
            for i in sorted_order
        ]
    
    def compare_faces(self, img1_path: str, img2_path: str) -> Dict:
        threshold_config = get_threshold_config()
        default_threshold = threshold_config.get("cosine_similarity", 0.5)
        
        emb1, _ = self.extract_embedding(img1_path)
        emb2, _ = self.extract_embedding(img2_path)
        
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        euclidean_dist = np.linalg.norm(emb1 - emb2)
        
        return {
            'cosine_similarity': float(cosine_sim),
            'similarity_percent': float((cosine_sim + 1) / 2 * 100),
            'euclidean_distance': float(euclidean_dist),
            'is_same': cosine_sim > default_threshold,
            'threshold': default_threshold
        }


class _LazyFaceService:
    _instance = None
    _init_failed = None

    def __getattr__(self, name):
        if self._init_failed:
            raise RuntimeError(f"FaceService initialization failed: {self._init_failed}")
        if self._instance is None:
            try:
                self._instance = FaceService()
            except Exception as e:
                self._init_failed = e
                raise RuntimeError(f"FaceService initialization failed: {e}")
        return getattr(self._instance, name)


face_service = _LazyFaceService()
