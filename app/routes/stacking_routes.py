import joblib
from flask import Blueprint, request, jsonify, current_app
import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.data_loader import dataloader
from app.core.model_predictor import ModelPredictor
from app.core.stacking_trainer import StackingTrainer
from app.models.stacking_training_record import StackingTrainingRecord
from app.models.stacking_prediction_record import StackingPredictionRecord
from app.models.file_model import UserFile
from app.extensions import db
from app.routes.predict_routes import save_result_file
from app.utils.jwt_utils import login_required

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
        model_id  = data.get('model_name')

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

        from app.core.data_loader import dataloader
        df = dataloader.load_file(data_path)
        X = df.drop(target, axis=1)
        y = df[target]

        # 4. 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 5. 初始化训练器
        trainer = StackingTrainer(task_type=task_type, cv=cross_validation)
        trainer.set_base_models(base_model_names)
        trainer.set_meta_model(meta_model_name)
        trainer.build_model()

        start_time = datetime.now()
        # 6. 训练模型
        trainer.train(X_train, y_train)
        end_time = datetime.now()

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
            start_time=start_time,
            end_time=end_time,
            metrics=metrics
        )

        db.session.add(record)
        db.session.commit()

        # 10. 返回成功响应
        return jsonify({
            'id': model_id,
            'model_name': model_id,
            'model_file_path': model_path,
            'model_file_size': os.path.getsize(model_path),
            'created_at': datetime.now(),
            'duration': (end_time - start_time).total_seconds(),
            'metrics': metrics,
        })

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(str(e))
        return jsonify({'error': str(e)}), 500


@stacking_bp.route('/predict', methods=['POST'])
@login_required
def predict_stacking_model():
    try:
        data = request.json

        # 1. 获取请求参数
        training_record_id = data.get('training_record_id')
        input_file_id = data.get('input_file_id')
        current_user = request.user

        if not training_record_id or not input_file_id:
            return jsonify({"error": "必须提供 training_record_id 和 input_file_id"}), 400

        # 2. 查询训练记录（通过 model_id）
        training_record = StackingTrainingRecord.query.get(training_record_id)
        if not training_record:
            return jsonify({"error": "未找到指定的训练记录"}), 404

        # 3. 获取模型信息
        target_column = training_record.target
        task_type = training_record.task_type

        # 获取输入文件记录
        user_file = UserFile.query.get(input_file_id)
        if not user_file:
            return jsonify({"error": "未找到指定的输入文件"}), 404

        model_path = training_record.model_path
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 404

        # 读取输入文件
        try:
            X = dataloader.load_file(user_file.file_path)
        except Exception as e:
            return jsonify({"error": f"读取文件失败: {e}"}), 500

        y_true = None
        if target_column in X.columns:
            y_true = X[target_column]
            X.drop(columns=[target_column], inplace=True)
        # 加载模型
        try:
            predictor = ModelPredictor(model_path, model_type=task_type)
        except Exception as e:
            return jsonify({"error": f"加载模型失败: {e}"}), 500

        # 执行预测
        try:
            start_time = datetime.utcnow()
            y_pred = predictor.predict(X)

            # 新增可视化数据生成
            visualization_data = predictor.generate_visualization_data(
                X, y_pred,
                y_true,
                target_column,
                task_type
            )
        except Exception as e:
            return jsonify({"error": f"预测失败: {e}"}), 500

        X['prediction'] = y_pred
        result_data = X.to_dict(orient='records')  # 转换为字典列表用于返回

        result_file_name = f"{user_file.file_name.split('.')[0]}_prediction" + str(
            int(datetime.utcnow().timestamp())) + "." + user_file.file_type
        try:
            result_file_path = save_result_file(X, user_file.user_id, result_file_name, user_file.file_type)
        except Exception as e:
            return jsonify({"error": f"保存预测结果文件失败: {e}"}), 500

        try:
            # 9. 构造预测记录并保存
            prediction_record = StackingPredictionRecord(
                training_record_id=training_record.id,
                input_file_id=input_file_id,
                user_id=current_user['user_id'],  # 从 UserFile 获取用户 ID
                start_time=start_time,
                end_time=datetime.now(),
                result_summary=visualization_data,
                result_path=result_file_path  # 示例路径
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
    except Exception as e:
        # 不再保存失败记录
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

    # current_app.logger.info(f"训练文件列：{expected_columns}")
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
    # current_app.logger.info(f"实际列：{actual_columns}")
    # 比较列
    missing = set(expected_columns) - set(actual_columns)
    extra = set(actual_columns) - set(expected_columns)
    # current_app.logger.info(f"列匹配结果：{missing} {extra}")
    return jsonify({
        "valid": len(missing) == 0,
        "missing_columns": list(missing),
        "extra_columns": list(extra),
        "expected_columns": expected_columns,
        "message": f"缺少{len(missing)}列，多余{len(extra)}列" if missing or extra else "列匹配"
    })