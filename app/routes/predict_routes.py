
from flask import Blueprint, request, jsonify
from app.extensions import db
from app.models.model_config import ModelConfig
from app.models.file_model import UserFile
from app.models.training_record import TrainingRecord
from app.models.predict_record import PredictRecord
from app.core.model_predictor import ModelPredictor
import pandas as pd
import json


predict_bp = Blueprint('predict', __name__, url_prefix='/predict')
@predict_bp.route('', methods=['POST'])
def predict():
    data = request.get_json()
    training_record_id = data.get('training_record_id')
    input_file_id = data.get('input_file_id')

    if not training_record_id or not input_file_id:
        return jsonify({"error": "必须提供 training_record_id 和 input_file_id"}), 400

    # 获取模型记录
    training_record = TrainingRecord.query.get(training_record_id)
    if not training_record:
        return jsonify({"error": "未找到指定的训练记录"}), 404

    # 获取模型配置（用于获取 category）
    model_config = ModelConfig.query.filter_by(model_type=training_record.model_type).first()
    if not model_config:
        return jsonify({"error": f"未找到模型类型 {training_record.model_type} 的配置"}), 500
    category = model_config.category  # 获取模型类别

    # 获取输入文件记录
    user_file = UserFile.query.get(input_file_id)
    if not user_file:
        return jsonify({"error": "未找到指定的输入文件"}), 404

    model_path = training_record.model_file_path

    # 读取输入文件
    try:
        X = pd.read_csv(user_file.file_path)
    except Exception as e:
        return jsonify({"error": f"读取文件失败: {e}"}), 500

    # 加载模型
    try:
        predictor = ModelPredictor(model_path, model_type=category)
    except Exception as e:
        return jsonify({"error": f"加载模型失败: {e}"}), 500

    # 执行预测
    try:
        y_pred = predictor.predict(X)
    except Exception as e:
        return jsonify({"error": f"预测失败: {e}"}), 500


    # 保存预测记录（可选：将 category 保存到 PredictRecord）

    # 返回响应
    return jsonify({
        # "prediction": prediction_record.to_dict(),
        "predict_data": y_pred.tolist(),
        # "category": category  # 可选：在响应中包含模型类别
    }), 200
