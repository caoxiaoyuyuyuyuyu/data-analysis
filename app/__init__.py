# backend/__init__.py
from flask import Flask
from flask_cors import CORS

from .extensions import db, migrate
from .config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    CORS(app, supports_credentials=True, origins=["http://localhost:5173"],
    allow_headers=["Authorization", "Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
    expose_headers=["Authorization"])

    # 初始化扩展
    db.init_app(app)
    migrate.init_app(app, db)

    # 注册蓝图
    from .routes.auth_routes import auth_bp
    from .routes.file_routes import file_bp  # 添加文件路由
    from .routes.home_routes import home_bp
    from .routes.preprocessing_routes import preprocessing_bp
    from .routes.history_routes import history_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(file_bp)  # 注册文件蓝图
    app.register_blueprint(home_bp)
    app.register_blueprint(preprocessing_bp)
    app.register_blueprint(history_bp)

    return app