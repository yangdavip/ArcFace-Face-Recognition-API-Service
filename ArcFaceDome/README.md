# ArcFace Face Recognition Demo

使用 ArcFace 模型实现的人脸识别比对演示项目。

## 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 项目结构

```
ArcFaceDome/
├── face_recognition.py    # 主程序
├── requirements.txt       # 依赖
├── test_images/          # 测试图片目录
│   ├── person1.jpg       # 人像1
│   ├── person2.jpg       # 人像2
│   └── target.jpg        # 比对目标
└── README.md
```

## 使用方法

### 1. 准备测试图片

在 `test_images` 目录下放入人脸图片:
- `person1.jpg` - 第一张人脸
- `person2.jpg` - 第二张人脸
- `target.jpg` - 待比对的目标人脸

### 2. 运行比对

```bash
python face_recognition.py
```

### 3. Python API

```python
from face_recognition import FaceRecognizer

recognizer = FaceRecognizer()

# 比对两张图片
result = recognizer.compare_faces('img1.jpg', 'img2.jpg')
print(result)
# {'cosine_similarity': 0.85, 'euclidean_distance': 1.2, 'is_same': True}

# 获取单张人脸特征
embedding = recognizer.get_face_embedding('person1.jpg')
```

### 相似度阈值

- Cosine Similarity > 0.5: 判定为同一人
- Euclidean Distance < 1.5: 判定为同一人

## 模型说明

使用 `buffalo_l` 预训练模型，基于 ArcFace 算法，支持:
- 人脸检测
- 人脸对齐
- 128维特征向量提取
- 余弦相似度/欧氏距离比对
