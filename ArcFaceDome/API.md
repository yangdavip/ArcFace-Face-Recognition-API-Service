# ArcFace 人脸识别 API 接口文档

## 简介

本系统基于 ArcFace 深度学习模型，提供完整的人脸识别能力，包括人脸库管理、人脸搜索、人脸检测等功能。

- **API 地址**: http://localhost:8000
- **交互式文档**: http://localhost:8000/docs
- **Redoc 文档**: http://localhost:8000/redoc

---

## 目录

1. [人脸库管理](#1-人脸库管理)
2. [库成员管理](#2-库成员管理)
3. [人脸搜索](#3-人脸搜索)
4. [人脸检测](#4-人脸检测)
5. [错误码说明](#5-错误码说明)

---

## 1. 人脸库管理

### 1.1 创建人脸库

创建一个新的人脸库，用于存储人脸特征向量。

**请求**

```http
POST /api/libraries
Content-Type: multipart/form-data
```

**参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| name | string | ✅ 是 | 人脸库名称，最大100字符 |
| description | string | ❌ 否 | 人脸库描述 |

**示例**

```bash
curl -X POST "http://localhost:8000/api/libraries" \
  -F "name=员工库" \
  -F "description=公司员工人脸数据库"
```

**响应 200**

```json
{
  "id": 1,
  "name": "员工库",
  "description": "公司员工人脸数据库",
  "created_at": "2026-02-27T10:00:00",
  "updated_at": "2026-02-27T10:00:00"
}
```

**响应 400** (名称已存在)

```json
{
  "detail": "Library name already exists"
}
```

---

### 1.2 获取人脸库列表

获取所有已创建的人脸库。

**请求**

```http
GET /api/libraries
```

**示例**

```bash
curl -X GET "http://localhost:8000/api/libraries"
```

**响应 200**

```json
[
  {
    "id": 1,
    "name": "员工库",
    "description": "公司员工人脸数据库",
    "created_at": "2026-02-27T10:00:00",
    "updated_at": "2026-02-27T10:00:00"
  },
  {
    "id": 2,
    "name": "访客库",
    "description": "访客人脸数据库",
    "created_at": "2026-02-27T11:00:00",
    "updated_at": "2026-02-27T11:00:00"
  }
]
```

---

### 1.3 获取人脸库详情

根据 ID 获取指定人脸库的详细信息。

**请求**

```http
GET /api/libraries/{library_id}
```

**参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| library_id | integer | ✅ 是 | 人脸库 ID |

**示例**

```bash
curl -X GET "http://localhost:8000/api/libraries/1"
```

**响应 200**

```json
{
  "id": 1,
  "name": "员工库",
  "description": "公司员工人脸数据库",
  "created_at": "2026-02-27T10:00:00",
  "updated_at": "2026-02-27T10:00:00"
}
```

**响应 404** (库不存在)

```json
{
  "detail": "Library not found"
}
```

---

### 1脸库

更新.4 修改人人脸库的名称或描述。

**请求**

```http
PUT /api/libraries/{library_id}
Content-Type: multipart/form-data
```

**参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| library_id | integer | ✅ 是 | 人脸库 ID |
| name | string | ❌ 否 | 新名称 |
| description | string | ❌ 否 | 新描述 |

**示例**

```bash
curl -X PUT "http://localhost:8000/api/libraries/1" \
  -F "name=新员工库"
```

**响应 200**

```json
{
  "id": 1,
  "name": "新员工库",
  "description": "公司员工人脸数据库",
  "created_at": "2026-02-27T10:00:00",
  "updated_at": "2026-02-27T12:00:00"
}
```

---

### 1.5 删除人脸库

删除指定的人脸库，同时删除该库下所有成员。

**请求**

```http
DELETE /api/libraries/{library_id}
```

**参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| library_id | integer | ✅ 是 | 人脸库 ID |

**示例**

```bash
curl -X DELETE "http://localhost:8000/api/libraries/1"
```

**响应 200**

```json
{
  "message": "Library deleted successfully"
}
```

---

## 2. 库成员管理

### 2.1 添加库成员 (文件上传)

上传人脸图片并添加到人脸库，系统自动提取特征向量。

**请求**

```http
POST /api/libraries/{library_id}/members
Content-Type: multipart/form-data
```

**参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| library_id | integer | ✅ 是 | 人脸库 ID |
| name | string | ✅ 是 | 成员姓名 |
| file | file | ✅ 是 | 人脸图片文件 |

**示例**

```bash
curl -X POST "http://localhost:8000/api/libraries/1/members" \
  -F "name=张三" \
  -F "file=@/path/to/face.jpg"
```

**响应 200**

```json
{
  "id": 1,
  "name": "张三",
  "image_path": "uploads/550e8400-e29b-41d4-a716-446655440000.jpg",
  "face_info": {
    "bbox": [120, 80, 280, 320],
    "landmarks": [
      [150, 120],
      [230, 120],
      [190, 180],
      [160, 230],
      [220, 230]
    ],
    "det_score": 0.9989
  },
  "created_at": "2026-02-27T10:00:00"
}
```

**响应 400** (未检测到人脸)

```json
{
  "detail": "Face extraction failed: No face detected in: uploads/xxx.jpg"
}
```

---

### 2.2 添加库成员 (文件路径)

通过本地文件路径添加成员到人脸库。

**请求**

```http
POST /api/libraries/{library_id}/members/by-path
Content-Type: application/x-www-form-urlencoded
```

**参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| library_id | integer | ✅ 是 | 人脸库 ID |
| name | string | ✅ 是 | 成员姓名 |
| image_path | string | ✅ 是 | 本地图片路径 |

**示例**

```bash
curl -X POST "http://localhost:8000/api/libraries/1/members/by-path" \
  -F "name=张三" \
  -F "image_path=test_images/person1.jpg"
```

**响应 200**

```json
{
  "id": 1,
  "name": "张三",
  "image_path": "test_images/person1.jpg",
  "face_info": {
    "bbox": [120, 80, 280, 320],
    "landmarks": [
      [150, 120],
      [230, 120],
      [190, 180],
      [160, 230],
      [220, 230]
    ],
    "det_score": 0.9989
  },
  "created_at": "2026-02-27T10:00:00"
}
```

---

### 2.3 分页查询库成员

获取指定人脸库中的所有成员，支持分页。

**请求**

```http
GET /api/libraries/{library_id}/members?page=1&page_size=10
```

**参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| library_id | integer | ✅ 是 | - | 人脸库 ID |
| page | integer | ❌ 否 | 1 | 页码，从1开始 |
| page_size | integer | ❌ 否 | 10 | 每页数量，范围1-100 |

**示例**

```bash
curl -X GET "http://localhost:8000/api/libraries/1/members?page=1&page_size=10"
```

**响应 200**

```json
{
  "total": 50,
  "page": 1,
  "page_size": 10,
  "items": [
    {
      "id": 1,
      "name": "张三",
      "image_path": "uploads/xxx.jpg",
      "created_at": "2026-02-27T10:00:00"
    },
    {
      "id": 2,
      "name": "李四",
      "image_path": "uploads/yyy.jpg",
      "created_at": "2026-02-27T10:30:00"
    }
  ]
}
```

---

### 2.4 更新库成员

更新成员的姓名或重新上传人脸图片。

**请求**

```http
PUT /api/libraries/{library_id}/members/{member_id}
Content-Type: multipart/form-data
```

**参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| library_id | integer | ✅ 是 | 人脸库 ID |
| member_id | integer | ✅ 是 | 成员 ID |
| name | string | ❌ 否 | 新姓名 |
| file | file | ❌ 否 | 新人脸图片 |

**示例**

```bash
# 只更新姓名
curl -X PUT "http://localhost:8000/api/libraries/1/members/1" \
  -F "name=张三丰"

# 同时更新姓名和图片
curl -X PUT "http://localhost:8000/api/libraries/1/members/1" \
  -F "name=张三丰" \
  -F "file=@/path/to/new_face.jpg"
```

**响应 200**

```json
{
  "id": 1,
  "name": "张三丰",
  "image_path": "uploads/new_xxx.jpg",
  "updated_at": "2026-02-27T12:00:00"
}
```

---

### 2.5 删除库成员

从人脸库中删除指定成员。

**请求**

```http
DELETE /api/libraries/{library_id}/members/{member_id}
```

**参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| library_id | integer | ✅ 是 | 人脸库 ID |
| member_id | integer | ✅ 是 | 成员 ID |

**示例**

```bash
curl -X DELETE "http://localhost:8000/api/libraries/1/members/1"
```

**响应 200**

```json
{
  "message": "Member deleted successfully"
}
```

---

## 3. 人脸搜索

### 3.1 人脸搜索

上传待检索人脸图片，在指定人脸库中查找相似的人脸。

**请求**

```http
POST /api/search
Content-Type: multipart/form-data
```

**参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| library_id | integer | ✅ 是 | - | 搜索目标库 ID |
| file | file | ✅ 是 | - | 待搜索人脸图片 |
| top_k | integer | ❌ 否 | 10 | 返回前 k 个结果 |
| threshold | float | ❌ 否 | 0.5 | 相似度阈值 |

**相似度阈值说明**

| 阈值范围 | 判定结果 |
|----------|----------|
| > 0.75 | 高度相似，同一人 |
| 0.5 - 0.75 | 中度相似，可能是同一人 |
| < 0.5 | 不相似，不同人 |

**示例**

```bash
curl -X POST "http://localhost:8000/api/search" \
  -F "library_id=1" \
  -F "file=@/path/to/search_face.jpg" \
  -F "top_k=5" \
  -F "threshold=0.5"
```

**响应 200**

```json
{
  "query_face": {
    "bbox": [120, 80, 280, 320],
    "landmarks": [
      [150, 120],
      [230, 120],
      [190, 180],
      [160, 230],
      [220, 230]
    ],
    "det_score": 0.9989
  },
  "results": [
    {
      "member_id": 1,
      "name": "张三",
      "similarity": 0.92,
      "similarity_percent": 96.0
    },
    {
      "member_id": 3,
      "name": "李四",
      "similarity": 0.65,
      "similarity_percent": 82.5
    }
  ]
}
```

**响应字段说明**

| 字段 | 说明 |
|------|------|
| query_face | 查询人脸的位置和关键点信息 |
| results | 搜索结果列表 |
| member_id | 匹配的成员 ID |
| name | 成员姓名 |
| similarity | 余弦相似度，范围 [-1, 1] |
| similarity_percent | 百分比形式，范围 [0, 100%] |

---

## 4. 人脸检测

### 4.1 人脸检测

检测图片中的人脸位置和关键点。

**请求**

```http
POST /api/detect
Content-Type: multipart/form-data
```

**参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | file | ✅ 是 | 待检测图片 |

**示例**

```bash
curl -X POST "http://localhost:8000/api/detect" \
  -F "file=@/path/to/image.jpg"
```

**响应 200**

```json
{
  "faces": [
    {
      "bbox": [120, 80, 280, 320],
      "landmarks": [
        [150, 120],
        [230, 120],
        [190, 180],
        [160, 230],
        [220, 230]
      ],
      "score": 0.9989
    }
  ],
  "count": 1
}
```

**响应字段说明**

| 字段 | 说明 |
|------|------|
| faces | 检测到的人脸列表 |
| bbox | 人脸边界框 [x1, y1, x2, y2] |
| landmarks | 5个关键点坐标 |
| score | 检测置信度 |

---

### 4.2 人脸关键点置信度检测

检测人脸并返回每个关键点的置信度。

**请求**

```http
POST /api/detect/confidence
Content-Type: multipart/form-data
```

**参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | file | ✅ 是 | 待检测图片 |

**示例**

```bash
curl -X POST "http://localhost:8000/api/detect/confidence" \
  -F "file=@/path/to/image.jpg"
```

**响应 200**

```json
{
  "faces": [
    {
      "bbox": [120, 80, 280, 320],
      "det_score": 0.9989,
      "landmarks": [
        {
          "point": [150, 120],
          "confidence": 0.99
        },
        {
          "point": [230, 120],
          "confidence": 0.98
        },
        {
          "point": [190, 180],
          "confidence": 0.97
        },
        {
          "point": [160, 230],
          "confidence": 0.96
        },
        {
          "point": [220, 230],
          "confidence": 0.95
        }
      ]
    }
  ],
  "count": 1
}
```

**5个关键点说明**

| 关键点 | 位置 |
|--------|------|
| 0 | 左眼 |
| 1 | 右眼 |
| 2 | 鼻子 |
| 3 | 左边嘴角 |
| 4 | 右边嘴角 |

---

## 5. 错误码说明

| HTTP 状态码 | 说明 |
|-------------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 422 | 数据验证失败 |
| 500 | 服务器内部错误 |

**常见错误信息**

| 错误信息 | 说明 | 解决方案 |
|----------|------|----------|
| Library not found | 人脸库不存在 | 检查 library_id 是否正确 |
| Member not found | 成员不存在 | 检查 member_id 是否正确 |
| No face detected | 未检测到人脸 | 确保图片中包含清晰的人脸 |
| Multiple faces detected | 检测到多个人脸 | 使用单人脸图片 |
| Library name already exists | 库名称已存在 | 使用不同的名称 |
| Image file not found | 图片文件不存在 | 检查文件路径是否正确 |

---

## 使用示例

### Python 请求示例

```python
import requests

# 1. 创建人脸库
resp = requests.post(
    "http://localhost:8000/api/libraries",
    data={"name": "员工库", "description": "测试库"}
)
library_id = resp.json()["id"]

# 2. 添加成员
with open("face.jpg", "rb") as f:
    resp = requests.post(
        f"http://localhost:8000/api/libraries/{library_id}/members",
        data={"name": "张三"},
        files={"file": f}
    )

# 3. 人脸搜索
with open("search.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/api/search",
        data={"library_id": library_id, "top_k": 5, "threshold": 0.5},
        files={"file": f}
    )
    print(resp.json())
```

### cURL 示例

```bash
# 创建人脸库
curl -X POST "http://localhost:8000/api/libraries" \
  -F "name=员工库"

# 添加成员
curl -X POST "http://localhost:8000/api/libraries/1/members" \
  -F "name=张三" \
  -F "file=@face.jpg"

# 人脸搜索
curl -X POST "http://localhost:8000/api/search" \
  -F "library_id=1" \
  -F "file=@search.jpg"

# 人脸检测
curl -X POST "http://localhost:8000/api/detect" \
  -F "file=@test.jpg"
```

---

## 技术栈

- **后端框架**: FastAPI
- **人脸识别**: InsightFace (ArcFace)
- **数据库**: PostgreSQL / SQLite
- **模型**: buffalo_l (512维特征向量)
