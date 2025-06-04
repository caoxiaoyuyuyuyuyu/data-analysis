from datetime import datetime
from pathlib import Path

from flask import Blueprint, request, jsonify, current_app

from app.core.data_loader import dataloader
from app.extensions import db
from app.models.model_config import ModelConfig
from app.models.file_model import UserFile
from app.models.training_record import TrainingRecord
from app.models.predict_record import PredictRecord, PredictStatus
from app.core.model_predictor import ModelPredictor
import pandas as pd
import json

predict_bp = Blueprint('predict', __name__, url_prefix='/predict')

@predict_bp.route('/check', methods=['POST'])
def check_file():
    data = request.get_json()
    file_id = data.get('file_id')
    model_id = data.get('model_id')

    # 获取模型训练时的特征列
    training_record = TrainingRecord.query.get(model_id)
    if not training_record:
        return jsonify({"error": "模型不存在"}), 404

    train_file = UserFile.query.get(training_record.file_id)
    if not train_file:
      return jsonify({"error": "文件不存在"}), 404

    train_df = dataloader.load_file(train_file.file_path)

    # expected_columns = train_df.columns.tolist() - [training_record.target_column]
    expected_columns = list(set(train_df.columns.tolist()) - {training_record.target_column})

    current_app.logger.info(f"训练文件列：{expected_columns}")
    # expected_columns = json.loads(training_record.feature_columns)  # 假设存储为JSON字符串

    # 获取文件的列
    user_file = UserFile.query.get(file_id)
    if not user_file:
        return jsonify({"error": "文件不存在"}), 404

    try:
        df = dataloader.load_file(user_file.file_path)

        actual_columns = df.columns.tolist()
    except Exception as e:
        return jsonify({"error": f"读取文件失败: {e}"}), 500
    current_app.logger.info(f"实际列：{actual_columns}")
    # 比较列
    missing = set(expected_columns) - set(actual_columns)
    extra = set(actual_columns) - set(expected_columns)
    current_app.logger.info(f"列匹配结果：{missing} {extra}")
    return jsonify({
        "valid": len(missing) == 0,
        "missing_columns": list(missing),
        "extra_columns": list(extra),
        "expected_columns": expected_columns,
        "message": f"缺少{len(missing)}列，多余{len(extra)}列" if missing or extra else "列匹配"
    })

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
    model_config = ModelConfig.query.filter_by(id=training_record.model_config_id).first()
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
        X = dataloader.load_file(user_file.file_path)
    except Exception as e:
        return jsonify({"error": f"读取文件失败: {e}"}), 500

    # 加载模型
    try:
        predictor = ModelPredictor(model_path, model_type=category)
    except Exception as e:
        return jsonify({"error": f"加载模型失败: {e}"}), 500

    # 执行预测
    try:
        start_time  = datetime.utcnow()
        y_pred = predictor.predict(X)

        # 新增可视化数据生成
        visualization_data = predictor.generate_visualization_data(
            X, y_pred,
            training_record.target_column,
            model_config.category
        )
    except Exception as e:
        return jsonify({"error": f"预测失败: {e}"}), 500

    X['prediction'] = y_pred  # 添加预测列为新的一列
    result_data = X.to_dict(orient='records')  # 转换为字典列表用于返回

    result_file_name = f"{user_file.file_name.split('.')[0]}_prediction" + str(int(datetime.utcnow().timestamp())) + "." + user_file.file_type
    try:
        result_file_path = save_result_file(X, user_file.user_id, result_file_name, user_file.file_type)
    except Exception as e:
        return jsonify({"error": f"保存预测结果文件失败: {e}"}), 500

    try:
        prediction_record = PredictRecord(
            user_id=training_record.user_id,
            training_record_id=training_record_id,
            input_file_id=input_file_id,
            output_file_path=result_file_path,
            predict_time = datetime.utcnow(),
            predict_duration = (datetime.utcnow() - start_time).total_seconds(),
            status='completed',
            error_message=None
        )
        db.session.add(prediction_record)
        db.session.commit()

        # 返回响应
        return jsonify({
            "prediction_record": prediction_record.to_dict(),
            "predict_data": result_data,
            "visualization": visualization_data  # 新增可视化数据
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"保存预测结果文件失败: {e}"}), 500

import json
def save_result_file(data, user_id, file_name, file_type):
    save_dir = Path(current_app.config['UPLOAD_FOLDER'], str(user_id), "predict")
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / file_name

    try:
        if file_type == 'csv':
            data.to_csv(file_path, index=False)
        elif file_type == 'xlsx':
            data.to_excel(file_path, index=False)
        elif file_type == 'txt':
            with open(file_path, 'w', encoding='utf-8') as f:
                for index, row in data.iterrows():
                    f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
    except Exception as e:
        current_app.logger.error(f"保存预测结果文件失败: {str(e)}")
        raise  # 继续抛出异常，由上层处理

    return str(file_path)

