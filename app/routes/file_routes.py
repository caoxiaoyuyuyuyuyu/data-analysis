from dataclasses import dataclass
from flask import Blueprint, request, jsonify, current_app
import os
from werkzeug.utils import secure_filename
from app.extensions import db
from app.models.file_model import UserFile
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
    is_processed: bool

file_bp = Blueprint('files', __name__, url_prefix='/files')

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}

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
        file.save(filepath)

        new_file = UserFile(
            user_id=current_user["user_id"],
            file_name=filename,
            file_path=filepath,
            file_size=os.path.getsize(filepath),
            file_type=file_type,
            description=description,
            is_processed=False
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
            is_processed=new_file.is_processed
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
            is_processed=f.is_processed
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
            is_processed=file.is_processed
        ))
    except Exception as e:
        current_app.logger.error(f"Get file error: {str(e)}")
        return jsonify({"error": "Failed to get file"}), 500