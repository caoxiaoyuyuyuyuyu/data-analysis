# app/routes/model_routes.py
import os

import joblib
import pandas as pd
from flask import Blueprint, jsonify, request, current_app
from sqlalchemy import text

from app.models.file_model import UserFile
from app.models.model_config import ModelConfig, ModelParameter
from app.extensions import db
from app.models.training_record import TrainingRecord
from app.utils.jwt_utils import login_required
from datetime import datetime

model_bp = Blueprint('model', __name__, url_prefix='/models')


@model_bp.route('/configs', methods=['GET'])
def get_model_configs():
    try:
        # 获取所有活跃的模型配置
        configs = ModelConfig.query.filter_by(is_active=True).all()

        result = []
        for config in configs:
            config_data = {
                'model_type': config.model_type,
                'display_name': config.display_name,
                'category': config.category,
                'description': config.description,
                'parameters': []
            }

            # 获取关联参数并按顺序排序
            parameters = ModelParameter.query.filter_by(
                model_config_id=config.id
            ).order_by(ModelParameter.param_order).all()

            for param in parameters:
                param_data = {
                    'name': param.name,
                    'type': param.type,
                    'default': param.default_value,
                    'description': param.description,
                    'required': param.is_required
                }

                # 根据类型添加额外属性
                if param.type == 'number':
                    param_data.update({
                        'min': float(param.min_value) if param.min_value is not None else None,
                        'max': float(param.max_value) if param.max_value is not None else None,
                        'step': float(param.step_size) if param.step_size is not None else None
                    })
                elif param.type == 'select':
                    param_data['options'] = param.options

                config_data['parameters'].append(param_data)

            result.append(config_data)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@model_bp.route('/configs', methods=['POST'])
@login_required
def create_model_config():
    try:
        data = request.get_json()

        # 验证必要字段
        if not all(key in data for key in ['model_type', 'display_name', 'category']):
            return jsonify({'error': 'Missing required fields'}), 400

        # 创建模型配置
        config = ModelConfig(
            model_type=data['model_type'],
            display_name=data['display_name'],
            category=data['category'],
            description=data.get('description'),
            is_active=data.get('is_active', True)
        )
        db.session.add(config)
        db.session.flush()  # 获取config.id

        # 添加参数
        for param_data in data.get('parameters', []):
            param = ModelParameter(
                model_config_id=config.id,
                name=param_data['name'],
                type=param_data['type'],
                default_value=param_data.get('default'),
                min_value=param_data.get('min'),
                max_value=param_data.get('max'),
                step_size=param_data.get('step'),
                options=param_data.get('options'),
                description=param_data.get('description'),
                param_order=param_data.get('order', 0),
                is_required=param_data.get('required', True)
            )
            db.session.add(param)

        db.session.commit()
        return jsonify({'message': 'Model config created successfully', 'id': config.id}), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@model_bp.route('/configs/<model_type>', methods=['PUT'])
@login_required
def update_model_config(model_type):
    try:
        config = ModelConfig.query.filter_by(model_type=model_type).first()
        if not config:
            return jsonify({'error': 'Model config not found'}), 404

        data = request.get_json()

        # 更新配置
        if 'display_name' in data:
            config.display_name = data['display_name']
        if 'category' in data:
            config.category = data['category']
        if 'description' in data:
            config.description = data['description']
        if 'is_active' in data:
            config.is_active = data['is_active']

        # 更新参数 (简化版，实际可能需要更复杂的合并逻辑)
        if 'parameters' in data:
            # 先删除旧参数
            ModelParameter.query.filter_by(model_config_id=config.id).delete()

            # 添加新参数
            for param_data in data['parameters']:
                param = ModelParameter(
                    model_config_id=config.id,
                    name=param_data['name'],
                    type=param_data['type'],
                    default_value=param_data.get('default'),
                    # ...其他字段
                )
                db.session.add(param)

        config.updated_at = datetime.utcnow()
        db.session.commit()
        return jsonify({'message': 'Model config updated successfully'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

from app.core.model_trainer import ModelTrainer
from app.core.data_loader import dataloader
@model_bp.route('/train', methods=['POST'])
@login_required
def train_model():
    try:
        data = request.get_json()
        current_user = request.user

        # 验证必要字段
        required_fields = ['file_id', 'target_column', 'model_config']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing required fields: {required_fields}'}), 400

        # 获取文件数据
        file = UserFile.query.get(data['file_id'])
        if not file:
            return jsonify({'error': 'File not found'}), 404

        # 读取文件数据
        df = dataloader.load_file(file.file_path)

        # 检查目标列是否存在
        if data['target_column'] not in df.columns:
            return jsonify({'error': f'Target column "{data["target_column"]}" not found in file'}), 400

        # 准备训练数据
        X = df.drop(columns=[data['target_column']])
        y = df[data['target_column']]

        # 初始化模型训练器
        trainer = ModelTrainer()

        # 获取模型配置
        model_type = data['model_config']['model_type']
        model_params = data['model_config'].get('parameters', {})
        test_size = data.get('test_size', 0.2)

        # 扩展模型类型映射
        model_name_map = {
            'Linear_Regression': 'Linear Regression',
            'Ridge_Regression': 'Ridge Regression',
            'Lasso_Regression': 'Lasso Regression',
            'Decision_Tree_Classifier': 'Decision Tree',
            'Decision_Tree_Regressor': 'Decision Tree',
            'Random_Forest_Classifier': 'Random Forest',
            'Random_Forest_Regressor': 'Random Forest',
            'SVM_Classifier': 'SVM',
            'SVM_Regressor': 'SVR',
            'KNN_Classifier': 'KNN Classification',
            'KNN_Regressor': 'KNN Regression',
            'Logistic_Regression': 'Logistic Regression',
            'Polynomial_Regression': 'Polynomial Regression'
        }

        model_name = model_name_map.get(model_type)
        if not model_name:
            return jsonify({'error': f'Unsupported model type: {model_type}'}), 400

        # 记录开始时间
        start_time = datetime.utcnow()

        # 训练模型
        result = trainer.train_model(
            X, y,
            model_name=model_name,
            test_size=test_size,
            **model_params
        )

        # 计算训练时长
        duration = (datetime.utcnow() - start_time).total_seconds()

        # 保存模型
        model_name_save = data.get('model_name', f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        model_path = save_model_to_file(result['model'], model_name_save, current_user['user_id'])

        # 创建模型记录
        new_model = TrainingRecord(
            user_id=current_user['user_id'],
            file_id=data['file_id'],
            file_name=file.file_name,
            model_name=model_name,
            duration=duration,
            metrics=result['metrics'],
            model_parameters=model_params,
            learning_curve=result['learning_curve'],
            created_at=datetime.utcnow(),
            updated_at = datetime.utcnow(),
            model_file_path=model_path,
            model_file_size=os.path.getsize(model_path),
        )
        db.session.add(new_model)
        db.session.commit()

        # 返回结果
        response = new_model.to_dict()

        return jsonify(response), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

def save_model_to_file(model, model_name,user_id):
    """保存模型到文件并返回路径"""
    model_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], str(user_id), "saved_models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    joblib.dump(model, model_path)

    return model_path