# ArcFace 人脸识别 API 服务

基于 ArcFace 深度学习模型的人脸识别 API 服务，支持人脸库管理、人脸搜索、人脸检测等功能。

## 功能特性

- 人脸库管理（创建、修改、删除、查询）
- 库成员管理（添加、修改、删除、分页查询）
- 人脸搜索（1:N 比对）
- 人脸检测与人脸关键点置信度检测
- 支持文件上传和 Base64 两种图片格式
- 支持 JSON 和 Form 两种请求格式

## 技术栈

- **后端框架**: FastAPI
- **人脸识别**: InsightFace (ArcFace)
- **数据库**: PostgreSQL / SQLite
- **模型**: buffalo_l (512维特征向量)

## 项目结构

```
ArcFaceDome/
├── main.py                 # FastAPI 主程序
├── database.py             # 数据库配置和模型
├── config_loader.py        # 配置加载器
├── face_service.py         # 人脸识别服务
├── warmup.py               # 模型预热
├── worker.py               # 线程池配置
├── settings.py             # 生产环境配置
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖
├── Dockerfile              # Docker 构建文件
├── docker-compose.yml      # Docker Compose 配置
├── uploads/                # 上传文件目录
└── README.md
```

## 配置文件说明

项目使用 `config.yaml` 配置文件，支持 SQLite 和 PostgreSQL 切换：

```yaml
# config.yaml
database:
  type: sqlite              # 数据库类型: sqlite / postgresql
  url: sqlite:///./face_recognition.db  # 数据库连接地址

server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  reload: false

model:
  name: buffalo_l
  det_size: [640, 640]

# 判定阈值配置
threshold:
  cosine_similarity: 0.5      # 余弦相似度阈值，>此值判定为同一人
  similarity_percent: 75       # 相似度百分比阈值

upload:
  max_file_size: 10485760  # 10MB
  allowed_extensions: [jpg, jpeg, png, bmp]
```

### 切换数据库

**使用 SQLite（默认）**:
```yaml
database:
  type: sqlite
  url: sqlite:///./face_recognition.db
```

**使用 PostgreSQL**:
```yaml
database:
  type: postgresql
  url: postgresql://user:password@localhost:5432/face_recognition
```

环境变量优先级高于配置文件：
```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/dbname"
```
 
默认数据库为 SQLite。若不设置 DATABASE_URL，系统将使用 sqlite:///./face_recognition.db。若要切换，请使用 DATABASE_URL 指定 PostgreSQL，或在 config.yaml 中将 database.type 设置为 postgresql，并配置 url。

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo_url>
cd ArcFaceDome

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置数据库

编辑 `config.yaml` 切换数据库类型：

```bash
# 使用 SQLite（默认）
vim config.yaml
# 设置 database.type: sqlite

# 或使用 PostgreSQL  
vim config.yaml
# 设置 database.type: postgresql
# 设置 database.url: postgresql://user:pass@localhost:5432/dbname
```

也可使用环境变量覆盖：
```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/dbname"
```

### 3. 启动服务

默认 GPU 加速启动
直接执行:
```bash
docker-compose up -d
```

服务启动后访问：
- API 文档: http://localhost:8000/docs
- ReDoc:    http://localhost:8000/redoc

## API 接口

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/libraries` | 创建人脸库 |
| GET | `/api/libraries` | 获取人脸库列表 |
| GET | `/api/libraries/{id}` | 获取人脸库详情 |
| PUT | `/api/libraries/{id}` | 修改人脸库 |
| DELETE | `/api/libraries/{id}` | 删除人脸库 |
| GET | `/api/libraries/{id}/members` | 分页查询库成员 |
| POST | `/api/libraries/{id}/members` | 添加库成员（文件上传）|
| POST | `/api/libraries/{id}/members/base64` | 添加库成员（Base64）|
| POST | `/api/libraries/{id}/members/by-path` | 添加库成员（文件路径）|
| PUT | `/api/libraries/{id}/members/{mid}` | 更新库成员 |
| DELETE | `/api/libraries/{id}/members/{mid}` | 删除库成员 |
| GET | `/api/libraries/{id}/members/by-record/{record_id}` | 根据record_id查询成员 |
| DELETE | `/api/libraries/{id}/members/by-record/{record_id}` | 根据record_id删除成员 |
| POST | `/api/search` | 人脸搜索（支持文件/Base64）|
| POST | `/api/search/json` | 人脸搜索（JSON格式）|
| POST | `/api/search/base64` | 人脸搜索（Base64格式）|
| POST | `/api/detect` | 人脸检测（文件上传）|
| POST | `/api/detect/base64` | 人脸检测（Base64格式）|
| POST | `/api/detect/confidence` | 人脸关键点置信度检测 |
| POST | `/api/detect/confidence/base64` | 人脸关键点置信度检测（Base64）|

## 请求示例

### 创建人脸库

```bash
curl -X POST "http://localhost:8000/api/libraries" \
  -H "Content-Type: application/json" \
  -d '{"name": "员工库", "description": "公司员工人脸库"}'
```

### 添加库成员（Base64）

```bash
curl -X POST "http://localhost:8000/api/libraries/1/members/base64" \
  -H "Content-Type: application/json" \
  -d '{"name": "张三", "image": "data:image/jpeg;base64,..."}'
```

### 人脸搜索

```bash
# JSON 格式
curl -X POST "http://localhost:8000/api/search/json" \
  -H "Content-Type: application/json" \
  -d '{"library_id": 1, "image": "data:image/jpeg;base64,...", "top_k": 10}'

# Form 格式（文件上传）
curl -X POST "http://localhost:8000/api/search?library_id=1" \
  -F "file=@face.jpg"
```

## 部署方式

### 方式一：直接部署（生产环境）

```bash
# 安装生产依赖
pip install gunicorn

# 使用 Gunicorn 启动（推荐）
gunicorn main:app -w 4 -b 0.0.0.0:8000 --worker-class uvicorn.workers.UvicornWorker

# 或使用 Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 方式二：Docker 部署

```bash
# 构建镜像
docker build -t arcface-api .

# 运行容器
docker run -d -p 8000:8000 \
  -e DATABASE_URL="postgresql://user:pass@host:5432/db" \
  arcface-api
```

### 方式三：Docker Compose 部署（推荐）

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 方式四：GPU 加速部署

```bash
# 安装 CUDA 版本的 onnxruntime
pip install onnxruntime-gpu

# 使用 GPU 运行
docker-compose -f docker-compose.gpu.yml up -d
```

## GPU 加速配置（默认）
在本仓库中，`docker-compose.yml` 已配置为使用 NVIDIA GPU 运行时，直接运行 `docker-compose up -d` 即可完成 GPU 加速部署。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATABASE_URL` | `sqlite:///./face_recognition.db` | 数据库连接地址 |
| `WORKERS` | 4 | 工作进程数 |
| `MAX_WORKERS` | 10 | 最大工作线程数 |
| `HOST` | 0.0.0.0 | 监听地址 |
| `PORT` | 8000 | 监听端口 |

## 相似度说明

- **Cosine Similarity**: 余弦相似度，范围 [-1, 1]
- **Similarity Percent**: 转换后的百分比，范围 [0, 100%]
- **判定阈值**: 可在 `config.yaml` 中配置

```yaml
threshold:
  cosine_similarity: 0.5      # 余弦相似度阈值，>此值判定为同一人
  similarity_percent: 75       # 相似度百分比阈值
```

响应中会返回当前使用的阈值：
```json
{
  "is_same": true,
  "threshold": 0.5
}
```

## 常见问题

### 1. 数据库连接失败

确保 PostgreSQL 服务已启动，或切换到 SQLite：

```bash
export DATABASE_URL="sqlite:///./face_recognition.db"
```

### 2. 首次启动慢

首次启动需要下载 ArcFace 模型（约 100MB），请耐心等待。

### 3. 内存占用高

可以减少 `WORKERS` 数量或在 `docker-compose.yml` 中限制内存。

## 许可证

MIT License
