# backend/routes/auth_routes.py
from dataclasses import dataclass
from flask import Blueprint, request, jsonify

from app.models.user_model import User  # 使用 ORM 模型
from app.utils.jwt_utils import generate_jwt, verify_jwt


@dataclass
class LoginRequest:
    email: str
    password: str

@dataclass
class RegisterRequest:
    username: str
    email: str
    password: str

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    req = LoginRequest(**data)

    user = User.find_by_email(req.email)
    if not user:
        return jsonify({"error": "User not found"}), 404

    if not user.is_active:
        return jsonify({"error": "User is inactive"}), 403

    if not user.check_password(req.password):
        return jsonify({"error": "Invalid password"}), 401

    user.update_last_login()

    token = generate_jwt(user.id, user.username, user.email)

    return jsonify({
        "token": token,
        "user": user.to_dict()
    })


@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    req = RegisterRequest(**data)

    if User.find_by_email(req.email):
        return jsonify({"error": "Email already exists"}), 400

    user = User(username=req.username, email=req.email)
    user.set_password(req.password)

    from app.extensions import db
    db.session.add(user)
    db.session.commit()

    token = generate_jwt(user.id, user.username, user.email)

    return jsonify({
        "token": token,
        "user": user.to_dict()
    })


@auth_bp.route('/me', methods=['GET'])
def get_me():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    payload = verify_jwt(token)
    if isinstance(payload, dict) and 'error' in payload:
        return jsonify(payload), 401

    user = User.find_by_email(payload['email'])
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify(user.to_dict())


@auth_bp.route('/profile', methods=['PUT'])
def update_profile():
    data = request.get_json()
    email = data.get('email')
    username = data.get('username')

    # 示例：假设是当前用户 ID 为 1
    user = User.find_by_id(1)
    if not user:
        return jsonify({"error": "User not found"}), 404

    if email:
        user.email = email
    if username:
        user.username = username

    from app.extensions import db
    db.session.commit()

    return jsonify({
        "id": user.id,
        "username": user.username,
        "email": user.email
    })
@auth_bp.route('/verify', methods=['GET'])
def verify_token():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return jsonify({"error": "Token is required"}), 401

    payload = verify_jwt(token)
    if isinstance(payload, dict) and 'error' in payload:
        return jsonify(payload), 401

    user = User.find_by_email(payload['email'])
    if not user:
        return jsonify({"error": "User not found"}), 404

    # 如果需要刷新 token 可以在这里生成新 token
    new_token = generate_jwt(user.id, user.username, user.email)

    return jsonify({
        "token": new_token,
        "user": user.to_dict()
    })
@auth_bp.route('/logout', methods=['POST'])
def logout():
    # 可以在这里添加黑名单等逻辑
    return jsonify({"message": "Logout successful"})