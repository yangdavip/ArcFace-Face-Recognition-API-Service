import os
import yaml
import threading
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = BASE_DIR / "config.yaml"

_config = None
_config_lock = threading.Lock()


def _get_config():
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                if CONFIG_FILE.exists():
                    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                        _config = yaml.safe_load(f) or {}
                else:
                    _config = {}
    return _config


def get_database_url():
    config = _get_config()
    db_config = config.get("database", {})
    db_type = db_config.get("type", "sqlite")
    
    if db_type == "sqlite":
        url = db_config.get("url", "sqlite:///./face_recognition.db")
        if url.startswith("sqlite:///./"):
            rel = url[len("sqlite:///./"):]
            resolved = BASE_DIR / rel
            return f"sqlite:///{resolved}"
        return url
    elif db_type == "postgresql":
        return os.environ.get("DATABASE_URL", db_config.get("url", ""))
    return f"sqlite:///{BASE_DIR / 'face_recognition.db'}"


def get_server_config():
    return _get_config().get("server", {})


def get_model_config():
    return _get_config().get("model", {})


def get_upload_config():
    return _get_config().get("upload", {})


def get_threshold_config():
    return _get_config().get("threshold", {
        "cosine_similarity": 0.5,
        "similarity_percent": 75
    })
