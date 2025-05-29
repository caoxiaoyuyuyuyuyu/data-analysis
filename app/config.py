import os
from datetime import timedelta

from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'your-flask-secret-key')

    # 数据库配置
    SQLALCHEMY_DATABASE_URI = (
        f"postgresql://{os.getenv('DB_USER', 'raguser')}:"
        f"{os.getenv('DB_PASSWORD', 'ragpassword')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'data_analysis')}"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 300,
        'pool_pre_ping': True
    }

    DEBUG = os.getenv('FLASK_DEBUG', 'False') == 'True'

    # 文件上传配置
    from pathlib import Path

    # 获取当前文件所在目录
    basedir = Path(__file__).resolve().parent

    UPLOAD_FOLDER = basedir.parent / 'uploads'

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}

    # JWT配置
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-super-secret-key')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_TOKEN_LOCATION = ['headers']
    JWT_HEADER_NAME = 'Authorization'
    JWT_HEADER_TYPE = 'Bearer'