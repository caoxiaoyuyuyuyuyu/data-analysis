# app/routes/model_routes.py
from flask import Blueprint, jsonify, request
from app.models.model_config import ModelConfig, ModelParameter
from app.extensions import db
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