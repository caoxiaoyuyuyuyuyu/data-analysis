from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import get_jwt_identity
from werkzeug.utils import secure_filename
import os
import uuid
import time
from datetime import datetime
import pandas as pd

from app.models.file_model import UserFile
from app.models.predict_record import PredictRecord, PredictStatus
from app.models.training_record import TrainingRecord
from app.extensions import db
from app.utils.jwt_utils import login_required

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'txt'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


predict_bp = Blueprint('predict', __name__, url_prefix='/predict')


@predict_bp.route('', methods=['POST'])
@login_required
def predict():
    """
    执行预测
    """
    # 验证请求数据
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    if 'model_id' not in request.form:
        return jsonify({"error": "No model_id provided"}), 400

    file = request.files['file']
    model_id = request.form['model_id']
    user_id = request.user['user_id']

    # 验证文件
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    # 获取模型信息
    training_record = TrainingRecord.query.filter_by(id=model_id, user_id=user_id).first()
    if not training_record:
        return jsonify({"error": "Model not found or not owned by user"}), 404

    # 创建预测记录
    predict_record = PredictRecord(
        user_id=user_id,
        training_record_id=training_record.id,
        model_type=training_record.model_type,
    )

    try:
        # 保存上传的文件
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"

        # 使用 os.path.join 构建完整路径
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], str(user_id), 'predict', unique_filename)

        # 创建目录
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 保存文件
        file.save(file_path)

        # 创建文件记录
        user_file = UserFile(
            user_id=user_id,
            file_name=filename,
            file_path=file_path,
            file_size=os.path.getsize(file_path),
            file_type=file.filename.rsplit('.', 1)[1]
        )
        db.session.add(user_file)
        db.session.flush()  # 获取file_id

        predict_record.input_file_id = user_file.id
        db.session.add(predict_record)
        db.session.commit()

        # 更新状态为处理中
        predict_record.status = PredictStatus.PROCESSING.value
        db.session.commit()

        start_time = time.time()

        from app.core.data_loader import dataloader
        df = dataloader.load_file(file_path)
        # 将数据转换为JSON格式存储
        input_data = df.to_dict(orient='records')
        predict_record.input_data = input_data
        db.session.commit()

        # TODO: 这里添加实际的预测逻辑
        # 使用训练好的模型进行预测
        # output_data = model.predict(df)

        # 模拟预测结果
        output_data = [{"prediction": "sample_result", "confidence": 0.95} for _ in range(len(df))]

        # 计算耗时
        duration = time.time() - start_time

        # 更新预测记录
        predict_record.update_result(
            output_data=output_data,
            duration=duration,
            status=PredictStatus.COMPLETED
        )
        db.session.commit()

        # 准备返回数据
        response_data = {
            "record_id": predict_record.id,
            "predictions": output_data,
            "model_info": {
                "id": training_record.id,
                "name": training_record.model_name,
                "type": training_record.model_type
            },
            "predict_time": datetime.utcnow().isoformat(),
            "predict_duration": duration
        }

        return jsonify(response_data), 200

    except Exception as e:
        current_app.logger.error(f"Prediction failed: {str(e)}")
        predict_record.update_result(
            output_data=None,
            duration=0,
            status=PredictStatus.FAILED,
            error_message=str(e)
        )
        db.session.commit()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500