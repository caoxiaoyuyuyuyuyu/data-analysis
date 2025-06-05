import joblib
from flask import Blueprint, request, jsonify
import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from app.core.stacking_trainer import StackingTrainer
from app.models.stacking_training_record import StackingTrainingRecord
from app.models.stacking_prediction_record import StackingPredictionRecord
from app.models.file_model import UserFile
from app.extensions import db

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


@stacking_bp.route('/predict', methods=['POST'])
def predict_stacking_model():
    try:
        data = request.json

        # 1. 获取请求参数
        model_id = data.get('model_id')
        file_id = data.get('file_id')

        if not model_id or not file_id:
            return jsonify({'error': 'Missing model_id or file_id'}), 400

        # 2. 查询训练记录（通过 model_id）
        record = StackingTrainingRecord.query.filter_by(model_id=model_id).first()
        if not record:
            return jsonify({'error': f"No training record found for model_id: {model_id}"}), 404

        # 3. 获取模型信息
        model_path = record.model_path
        target_column = record.target
        task_type = record.task_type

        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 404

        # 4. 加载模型
        model = joblib.load(model_path)

        # 5. 加载预测数据
        try:
            data_path = get_data_path_from_id(file_id)
        except ValueError as e:
            return jsonify({'error': str(e)}), 404

        if not os.path.exists(data_path):
            return jsonify({'error': 'Data file not found at the specified path'}), 404

        df = pd.read_csv(data_path)

        # 6. 数据预处理（与训练一致）
        if target_column in df.columns:
            X = df.drop(target_column, axis=1)
        else:
            X = df.copy()

        # 7. 执行预测
        predictions = model.predict(X)

        # 8. 构建结果摘要
        result_summary = {
            'total_predictions': len(predictions),
            'task_type': task_type,
            'target_column': target_column
        }

        # 9. 构造预测记录并保存
        prediction_record = StackingPredictionRecord(
            training_record_id=record.id,
            input_file_id=file_id,
            user_id=record.file.user_id,  # 从 UserFile 获取用户 ID
            start_time=datetime.now(),
            end_time=datetime.now(),
            result_summary=result_summary,
            result_path=f'/predictions/{model_id}_{file_id}.csv'  # 示例路径
        )

        db.session.add(prediction_record)
        db.session.commit()

        # 10. 返回结果
        return jsonify({
            'task_type': task_type,
            'target_column': target_column,
            'predictions': predictions.tolist()
        })

    except Exception as e:
        # 不再保存失败记录
        return jsonify({'error': str(e)}), 500