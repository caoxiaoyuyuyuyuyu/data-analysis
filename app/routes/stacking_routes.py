import joblib
from flask import Blueprint, request, jsonify, current_app
from pathlib import Path
import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from app.core.stacking_trainer import StackingTrainer
from app.core.model_predictor import ModelPredictor
from app.models.stacking_training_record import StackingTrainingRecord
from app.models.stacking_prediction_record import StackingPredictionRecord
from app.models.file_model import UserFile
from app.extensions import db
from app.core.data_loader import dataloader

stacking_bp = Blueprint('stacking', __name__, url_prefix='/stacking')

def get_data_path_from_id(file_id):
    """
    根据 file_id 查询 user_files 表获取 file_path。
    如果找不到对应记录，抛出 ValueError 异常。
    """
    user_file = UserFile.query.get(file_id)
    if not user_file:
        raise ValueError(f"File with id {file_id} not found in database.")
    return user_file.file_path


@stacking_bp.route('/train', methods=['POST'])
def train_stacking_model():
    try:
        data = request.json

        # 1. 解析请求参数
        input_file_id = data.get('input_file_id')
        base_model_names = data.get('base_model_name', [])
        meta_model_name = data.get('meta_model_name')
        task_type = data.get('task_type', 'classification')
        cross_validation = data.get('cross_validation', 5)
        target = data.get('target')

        # 2. 参数校验
        if not input_file_id:
            return jsonify({'error': 'Missing input_file_id'}), 400
        if not base_model_names:
            return jsonify({'error': 'At least one base model required'}), 400
        if not meta_model_name:
            return jsonify({'error': 'Meta model not specified'}), 400

        # 从数据库中获取文件路径
        try:
            data_path = get_data_path_from_id(input_file_id)
        except ValueError as e:
            return jsonify({'error': str(e)}), 404

        df = pd.read_csv(data_path)
        X = df.drop(target, axis=1)
        y = df[target]

        # 4. 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 5. 初始化训练器
        trainer = StackingTrainer(task_type=task_type, cv=cross_validation)
        trainer.set_base_models(base_model_names)
        trainer.set_meta_model(meta_model_name)
        trainer.build_model()

        # 6. 训练模型
        trainer.train(X_train, y_train)

        # 7. 评估模型
        metrics = trainer.evaluate(X_test, y_test)

        # 8. 保存模型
        model_id, model_path = trainer.save_model()

        # 9. 构造训练记录并保存
        record = StackingTrainingRecord(
            input_file_id=input_file_id,
            task_type=task_type,
            base_model_names=base_model_names,
            meta_model_name=meta_model_name,
            cross_validation=cross_validation,
            target=target,
            model_id=model_id,
            model_path=model_path,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=metrics
        )

        db.session.add(record)
        db.session.commit()

        # 10. 返回成功响应
        return jsonify({
            'model_id': model_id,
            'metrics': metrics
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@stacking_bp.route('/check', methods=['POST'])
def check_file():
    data = request.get_json()
    file_id = data.get('file_id')
    model_id = data.get('model_id')

    # 获取模型训练时的特征列
    training_record = StackingTrainingRecord.query.get(model_id)
    if not training_record:
        return jsonify({"error": "模型不存在"}), 404

    train_file = UserFile.query.get(training_record.input_file_id)
    if not train_file:
        return jsonify({"error": "文件不存在"}), 404

    train_df = dataloader.load_file(train_file.file_path)

    # expected_columns = train_df.columns.tolist() - [training_record.target_column]
    expected_columns = list(set(train_df.columns.tolist()) - {training_record.target})

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


@stacking_bp.route('/predict', methods=['POST'])
def predict_stacking_model():
    data = request.get_json()
    training_record_id = data.get('training_record_id')
    input_file_id = data.get('input_file_id')

    if not training_record_id or not input_file_id:
        return jsonify({"error": "必须提供 training_record_id 和 input_file_id"}), 400

    # 获取模型记录
    training_record = StackingTrainingRecord.query.get(training_record_id)
    if not training_record:
        return jsonify({"error": "未找到指定的训练记录"}), 404

    # 获取模型配置（用于获取 category）
    category = training_record.task_type  # 获取模型类别

    # 获取输入文件记录
    user_file = UserFile.query.get(input_file_id)
    if not user_file:
        return jsonify({"error": "未找到指定的输入文件"}), 404

    model_path = training_record.model_path

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
            training_record.target,
            category
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
        stacking_prediction_record = StackingPredictionRecord(
            user_id=user_file.user_id,
            training_record_id=training_record_id,
            input_file_id=input_file_id,
            start_time = start_time,
            end_time = datetime.utcnow(),
            result_path = result_file_path,
            result_summary = 'complete.'
        )
        db.session.add(stacking_prediction_record)
        db.session.commit()

        # 返回响应
        return jsonify({
            "prediction_record": stacking_prediction_record.to_dict(),
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

