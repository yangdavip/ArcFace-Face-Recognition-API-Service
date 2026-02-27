import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from pathlib import Path


class FaceRecognizer:
    def __init__(self, model_name='buffalo_l', providers=['CPUExecutionProvider']):
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def get_face_embedding(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError(f"No face detected in: {image_path}")
        if len(faces) > 1:
            raise ValueError(f"Multiple faces detected in: {image_path}")
        
        return faces[0].embedding
    
    @staticmethod
    def similarity_to_percent(cosine_sim):
        percent = (cosine_sim + 1) / 2 * 100
        return min(max(percent, 0), 100)
    
    def compare_faces(self, img1_path, img2_path):
        emb1 = self.get_face_embedding(img1_path)
        emb2 = self.get_face_embedding(img2_path)
        
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        euclidean_dist = np.linalg.norm(emb1 - emb2)
        
        return {
            'cosine_similarity': float(cosine_sim),
            'similarity_percent': self.similarity_to_percent(cosine_sim),
            'euclidean_distance': float(euclidean_dist),
            'is_same': cosine_sim > 0.5
        }


def demo_single_comparison():
    recognizer = FaceRecognizer()
    
    test_images = Path('test_images')
    img1 = test_images / 'person1.jpg'
    img2 = test_images / 'person2.jpg'
    
    if not img1.exists() or not img2.exists():
        print("Please put test images in test_images folder")
        print("Required: person1.jpg, person2.jpg")
        return
    
    result = recognizer.compare_faces(img1, img2)
    print(f"Cosine Similarity: {result['cosine_similarity']:.4f}")
    print(f"Similarity: {result['similarity_percent']:.2f}%")
    print(f"Euclidean Distance: {result['euclidean_distance']:.4f}")
    print(f"Same Person: {result['is_same']}")


def demo_batch_comparison():
    recognizer = FaceRecognizer()
    test_images = Path('test_images')
    
    gallery = {}
    for img_path in test_images.glob('*.jpg'):
        if img_path.stem.startswith('person'):
            emb = recognizer.get_face_embedding(img_path)
            gallery[img_path.stem] = emb
    
    print("\nGallery faces:", list(gallery.keys()))
    
    target = test_images / 'target.jpg'
    if not target.exists():
        print("Please put target.jpg in test_images folder")
        return
    
    target_emb = recognizer.get_face_embedding(target)
    
    print("\nComparison results:")
    for name, emb in gallery.items():
        sim = np.dot(target_emb, emb) / (np.linalg.norm(target_emb) * np.linalg.norm(emb))
        print(f"  {name}: {sim:.4f}")


if __name__ == '__main__':
    print("ArcFace Face Recognition Demo")
    print("=" * 40)
    
    demo_single_comparison()
