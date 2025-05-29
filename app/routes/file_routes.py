import json
import uuid
from dataclasses import dataclass
from flask import Blueprint, request, jsonify, current_app, send_file
import os
from werkzeug.utils import secure_filename
from app.extensions import db
from app.models.file_model import UserFile
from app.models.training_record import TrainingRecord
from app.utils.jwt_utils import login_required


@dataclass
class UserFileResponse:
    id: int
    user_id: int
    file_name: str
    file_size: int
    file_type: str
    upload_time: str
    description: str
    parent_id: bool

file_bp = Blueprint('files', __name__, url_prefix='/files')

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'txt'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@file_bp.route('/upload', methods=['POST'])
@login_required
def upload_file():
    current_user = request.user
    print(current_user)

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    description = request.form.get('description', '')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    def custom_secure_filename(filename):
        # 保留原始文件名，但移除可能导致问题的字符
        return filename.replace('/', '_').strip()

    try:
        filename = custom_secure_filename(file.filename)
        print(f"Uploading file: {filename}")
        file_type = filename.rsplit('.', 1)[1].lower()
        print(f"File type: {file_type}")
        upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], str(current_user["user_id"]))
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        current_app.logger.info(f"File uploaded: {filepath}")
        file.save(filepath)

        new_file = UserFile(
            user_id=current_user["user_id"],
            file_name=filename,
            file_path=filepath,
            file_size=os.path.getsize(filepath),
            file_type=file_type,
            description=description
        )
        db.session.add(new_file)
        db.session.commit()

        return jsonify(UserFileResponse(
            id=new_file.id,
            user_id=new_file.user_id,
            file_name=new_file.file_name,
            file_size=new_file.file_size,
            file_type=new_file.file_type,
            upload_time=new_file.upload_time.isoformat(),
            description=new_file.description,
            parent_id=new_file.parent_id
        ))

    except Exception as e:
        current_app.logger.error(f"File upload error: {str(e)}")
        return jsonify({"error": "File upload failed"}), 500

@file_bp.route('', methods=['GET'])
@login_required
def get_files():
    try:
        current_user = request.user
        print(current_user)
        files = UserFile.query.filter_by(
            user_id=current_user["user_id"]
        ).order_by(UserFile.upload_time.desc()).all()

        return jsonify([UserFileResponse(
            id=f.id,
            user_id=f.user_id,
            file_name=f.file_name,
            file_size=f.file_size,
            file_type=f.file_type,
            upload_time=f.upload_time.isoformat(),
            description=f.description,
            parent_id=f.parent_id
        ) for f in files])
    except Exception as e:
        current_app.logger.error(f"Get files error: {str(e)}")
        return jsonify({"error": "Failed to get files"}), 500


@file_bp.route('/<int:file_id>', methods=['GET'])
@login_required
def get_single_file(file_id):
    try:
        current_user = request.user
        file = UserFile.query.filter_by(
            user_id=current_user["user_id"],
            id=file_id
        ).first()

        if not file:
            return jsonify({"error": "File not found"}), 404

        return jsonify(UserFileResponse(
            id=file.id,
            user_id=file.user_id,
            file_name=file.file_name,
            file_size=file.file_size,
            file_type=file.file_type,
            upload_time=file.upload_time.isoformat(),
            description=file.description,
            parent_id=file.parent_id
        ))
    except Exception as e:
        current_app.logger.error(f"Get file error: {str(e)}")
        return jsonify({"error": "Failed to get file"}), 500

@file_bp.route('/<int:file_id>', methods=['DELETE'])
@login_required
def delete_file(file_id):
    try:
        current_user = request.user
        file = UserFile.query.filter_by(
            user_id=current_user["user_id"],
            id=file_id
        ).first()

        file_path = file.file_path
        if os.path.exists(file_path):
            os.remove(file_path)

        if not file:
            return jsonify({"error": "File not found"}), 404

        db.session.delete(file)
        db.session.commit()

        return jsonify({"message": "File deleted successfully"})
    except Exception as e:
        current_app.logger.error(f"Delete file error: {str(e)}")
        return jsonify({"error": "Failed to delete file"}), 500

@file_bp.route('/<int:file_id>/download', methods=['GET'])
@login_required
def download_file(file_id):
    try:
        current_user = request.user
        file = UserFile.query.filter_by(
            user_id=current_user["user_id"],
            id=file_id
        ).first()
        if not file:
            return jsonify({"error": "File not found"}), 404

        return send_file(file.file_path, as_attachment=True, download_name=file.file_name)
    except Exception as e:
        current_app.logger.error(f"Download file error: {str(e)}")
        return jsonify({"error": "Failed to download file"}), 500

# 文件检查路由
@file_bp.route('/check', methods=['POST'])
@login_required
def check_file():
    """检查文件是否符合要求"""
    try:
        current_user = request.user

        # 检查文件上传
        if 'file' not in request.files:
            return jsonify({
                "valid": False,
                "error": "No file part",
                "code": "NO_FILE"
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "valid": False,
                "error": "No selected file",
                "code": "EMPTY_FILE"
            }), 400

        # 检查文件扩展名
        if not allowed_file(file.filename):
            return jsonify({
                "valid": False,
                "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
                "code": "INVALID_TYPE"
            }), 400

        # 保存临时文件
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        temp_filename = f"check_{uuid.uuid4().hex}.{file_extension}"
        temp_file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_file_path)

        from app.core.data_loader import dataloader
        # 尝试加载文件
        df = dataloader.load_file(temp_file_path)
        if df is None:
            os.remove(temp_file_path)
            return jsonify({
                "valid": False,
                "error": "Failed to load file. Please check file format.",
                "code": "LOAD_FAILED"
            }), 400

        # 基础统计信息
        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "sample_data": df.head(5).to_dict(orient='records')
        }

        # 如果有模型ID，检查特征列
        model_id = request.form.get('model_id')
        if model_id:
            model = TrainingRecord.query.filter_by(
                id=model_id,
                user_id=current_user["user_id"]
            ).first()

            file = UserFile.query.filter_by(
                id=model.file_id,
                user_id=current_user["user_id"]
            ).first()
            if file:
                train_df = dataloader.load_file(file.file_path)
                feature_columns = [col for col in train_df.columns if col != model.target_column]
            else:
                return jsonify({
                    "valid": False,
                    "error": "Failed to load training file.",
                    "code": "LOAD_FAILED"
                }), 400

            if model and feature_columns:
                missing_cols = set(feature_columns) - set(df.columns)
                if missing_cols:
                    os.remove(temp_file_path)
                    return jsonify({
                        "valid": False,
                        "error": f"Missing columns: {', '.join(missing_cols)}",
                        "code": "MISSING_COLUMNS",
                        "missing_columns": list(missing_cols),
                        "required_columns": feature_columns
                    }), 400

                stats["feature_columns"] = feature_columns


        # 检查通过
        os.remove(temp_file_path)
        return jsonify({
            "valid": True,
            "message": "File is valid",
            "stats": stats
        })

    except Exception as e:
        current_app.logger.error(f"File check error: {str(e)}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return jsonify({
            "valid": False,
            "error": "Internal server error",
            "code": "SERVER_ERROR"
        }), 500