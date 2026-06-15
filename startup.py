#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

# 中文日志颜色
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def log_info(msg):
    print(f"{Colors.GREEN}[启动]{Colors.RESET} {msg}")

def log_warn(msg):
    print(f"{Colors.YELLOW}[警告]{Colors.RESET} {msg}")

def log_error(msg):
    print(f"{Colors.RED}[错误]{Colors.RESET} {msg}")

def log_success(msg):
    print(f"{Colors.GREEN}[成功]{Colors.RESET} {msg}")

def log_step(msg):
    print(f"{Colors.BLUE}[步骤]{Colors.RESET} {msg}")

def check_python_version():
    """检查 Python 版本"""
    log_step("检查 Python 版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        log_success(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        log_error(f"Python 版本过低: {version.major}.{version.minor}, 需要 3.8+")
        return False

def check_dependencies():
    """检查依赖包是否安装"""
    log_step("检查依赖包...")
    
    # Note: use concrete import names that match installed packages
    # - OpenCV for Python is imported as 'cv2' when using opencv-python
    required_packages = [
        'fastapi', 'uvicorn', 'sqlalchemy', 'insightface',
        'numpy', 'cv2', 'PIL', 'yaml',
    ]
    optional_packages = {
        'psycopg2': 'PostgreSQL 数据库支持',
    }
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    for package, desc in optional_packages.items():
        try:
            __import__(package)
        except ImportError:
            log_warn(f"缺少可选依赖 {package}（{desc}，当前使用 SQLite 可忽略）")
    
    if missing:
        log_error(f"缺少必要依赖包: {', '.join(missing)}")
        log_info("请运行: pip install -r requirements.txt")
        return False
    
    log_success(f"所有必要依赖包已安装")
    return True

def check_disk_space():
    """检查磁盘空间"""
    log_step("检查磁盘空间...")
    
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (2**30)
    
    if free_gb < 1:
        log_error(f"磁盘空间不足: 剩余 {free_gb} GB")
        return False
    elif free_gb < 5:
        log_warn(f"磁盘空间较小: 剩余 {free_gb} GB")
    else:
        log_success(f"磁盘空间充足: 剩余 {free_gb} GB")
    
    return True

def check_directories():
    """检查必要目录"""
    log_step("检查目录结构...")
    
    dirs = ['uploads', 'logs']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    
    log_success("目录检查完成")
    return True

def wait_for_database(max_retries=30, retry_interval=2):
    """等待数据库就绪"""
    log_step("等待数据库就绪...")
    
    from config_loader import get_database_url
    from sqlalchemy import create_engine, text
    
    db_url = os.getenv("DATABASE_URL") or get_database_url()
    
    for i in range(max_retries):
        try:
            engine = create_engine(db_url, echo=False)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log_success("数据库已就绪")
            return True
        except Exception as e:
            if i < max_retries - 1:
                log_warn(f"等待数据库连接... ({i+1}/{max_retries})")
                time.sleep(retry_interval)
            else:
                log_error(f"数据库连接超时: {str(e)}")
                return False
    
    return False

def init_database():
    """初始化数据库表
    使用 Inspector 检查表是否已创建，兼容 SQLite 和 PostgreSQL。
    """
    log_step("初始化数据库表...")
    try:
        from database import init_db, engine
        from sqlalchemy import inspect

        init_db()

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        required = {"face_libraries", "face_members"}
        if required.issubset(set(tables)):
            log_success(f"数据库表初始化成功: {', '.join(tables)}")
            return True
        else:
            log_error(f"数据库表初始化失败，当前表: {', '.join(tables)}")
            return False
    except Exception as e:
        log_error(f"数据库初始化异常: {str(e)}")
        return False

def check_model_files():
    """检查模型文件"""
    log_step("检查人脸识别模型...")
    
    try:
        from insightface.app import FaceAnalysis
        import os
        
        # 模型会自动下载到 ~/.insightface/models/
        model_path = Path.home() / ".insightface" / "models"
        
        log_info(f"模型路径: {model_path}")
        log_success("模型检查完成（首次使用会自动下载）")
        return True
    except Exception as e:
        log_warn(f"模型检查警告: {str(e)}")
        return True

def start_server():
    """启动服务"""
    log_step("启动 FastAPI 服务...")
    
    from database import DATABASE_URL

    import uvicorn
    
    workers = 1
    if DATABASE_URL and not DATABASE_URL.startswith("sqlite"):
        workers = 4
    else:
        log_warn("SQLite 模式 — 强制 workers=1 避免数据库锁冲突。生产环境建议使用 PostgreSQL。")
    
    log_success("服务启动成功!")
    log_info("=" * 50)
    log_info(f"Workers: {workers}")
    log_info("API 文档: http://localhost:8000/docs")
    log_info("ReDoc:    http://localhost:8000/redoc")
    log_info("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=workers,
        log_level="info"
    )

def main():
    print()
    log_info("=" * 50)
    log_info("  ArcFace 人脸识别 API 服务")
    log_info("=" * 50)
    print()
    
    # 检查 Python 版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查目录
    if not check_directories():
        sys.exit(1)
    
    # 检查磁盘空间
    if not check_disk_space():
        sys.exit(1)
    
    # 等待数据库就绪
    if not wait_for_database():
        sys.exit(1)
    
    # 初始化数据库
    if not init_database():
        sys.exit(1)
    
    # 检查模型
    if not check_model_files():
        sys.exit(1)
    
    print()
    log_success("所有检查通过！启动服务...")
    print()
    
    # 启动服务
    start_server()

if __name__ == "__main__":
    main()
