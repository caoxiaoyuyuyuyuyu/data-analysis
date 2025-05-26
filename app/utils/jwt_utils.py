# backend/utils/jwt_utils.py
import jwt
import datetime
import os
from functools import wraps
from flask import request, jsonify, current_app

def get_secret_key():
    return os.getenv('JWT_SECRET_KEY') or current_app.config.get('SECRET_KEY', 'fallback-secret-key')

def generate_jwt(user_id: int, username: str, email: str) -> str:
    payload = {
        'user_id': user_id,
        'username': username,
        'email': email,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    secret_key = get_secret_key()
    return jwt.encode(payload, secret_key, algorithm='HS256')

def verify_jwt(token: str):
    secret_key = get_secret_key()
    try:
        return jwt.decode(token, secret_key, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        return {'error': 'Token expired'}
    except jwt.InvalidTokenError:
        return {'error': 'Invalid token'}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 打印所有请求头（调试用）
        # current_app.logger.debug(f"所有请求头: {request.headers}")
        auth_header = request.headers.get('Authorization', '')
        # current_app.logger.debug(f"原始Authorization头: {auth_header}")

        # 提取Token（兼容Bearer前缀和直接Token）
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ', 1)[1].strip()
        else:
            token = auth_header.strip()

        current_app.logger.debug(f"提取的Token: {token}")

        if not token:
            return jsonify({'error': 'Missing token'}), 401
        payload = verify_jwt(token)
        if 'error' in payload:
            return jsonify({'error': payload['error']}), 401
        request.user = payload
        return f(*args, **kwargs)
    return decorated_function
