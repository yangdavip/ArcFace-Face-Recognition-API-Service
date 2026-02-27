import cv2
import numpy as np
from typing import List, Dict, Tuple
from insightface.app import FaceAnalysis


class FaceService:
    def __init__(self, model_name='buffalo_l', providers=['CPUExecutionProvider']):
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
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
            raise ValueError(f"Cannot read image: {image_path}")
        
        faces = self.app.get(img)
        results = []
        
        for face in faces:
            face_info = {
                'bbox': face.bbox.tolist(),
                'det_score': float(face.det_score) if hasattr(face, 'det_score') else 1.0,
            }
            
            if hasattr(face, 'kps') and face.kps is not None:
                landmarks = face.kps
                landmark_confidences = []
                for i in range(5):
                    x, y = landmarks[i]
                    landmark_confidences.append({
                        'point': [float(x), float(y)],
                        'confidence': float(face.det_score) * (1.0 - i * 0.05)
                    })
                face_info['landmarks'] = landmark_confidences
            else:
                face_info['landmarks'] = []
            
            results.append(face_info)
        
        return results
    
    def extract_embedding(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError(f"No face detected in: {image_path}")
        if len(faces) > 1:
            raise ValueError(f"Multiple faces detected in: {image_path}")
        
        face = faces[0]
        embedding = face.embedding
        
        return embedding, {
            'bbox': face.bbox.tolist(),
            'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
            'det_score': float(face.det_score) if hasattr(face, 'det_score') else 1.0,
        }
    
    def search_faces(self, query_embedding: np.ndarray, gallery_embeddings: List[Tuple[int, np.ndarray, str]], top_k: int = 10, threshold: float = 0.5) -> List[Dict]:
        results = []
        
        for member_id, emb, name in gallery_embeddings:
            cosine_sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            
            if cosine_sim >= threshold:
                results.append({
                    'member_id': member_id,
                    'name': name,
                    'similarity': float(cosine_sim),
                    'similarity_percent': float((cosine_sim + 1) / 2 * 100),
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def compare_faces(self, img1_path: str, img2_path: str) -> Dict:
        emb1, _ = self.extract_embedding(img1_path)
        emb2, _ = self.extract_embedding(img2_path)
        
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        euclidean_dist = np.linalg.norm(emb1 - emb2)
        
        return {
            'cosine_similarity': float(cosine_sim),
            'similarity_percent': float((cosine_sim + 1) / 2 * 100),
            'euclidean_distance': float(euclidean_dist),
            'is_same': cosine_sim > 0.5
        }


face_service = FaceService()
