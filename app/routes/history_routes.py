# history_routes.py
from flask import Blueprint, jsonify, request, current_app

from app.models.preprocessing_step import PreprocessingStep
from app.models.stacking_prediction_record import StackingPredictionRecord
from app.models.stacking_training_record import StackingTrainingRecord
from app.models.training_record import TrainingRecord
from app.models.predict_record import PredictRecord
from app.models.preprocessing_record import PreprocessingRecord
from app.utils.jwt_utils import login_required
from app.extensions import db
from datetime import datetime

history_bp = Blueprint('history', __name__, url_prefix='/history')


# 训练历史记录路由
@history_bp.route('/training', methods=['GET'])
@login_required
def get_training_history():
    try:
        current_user = request.user
        records = TrainingRecord.query.filter_by(
            user_id=current_user["user_id"]
        ).order_by(TrainingRecord.created_at.desc()).all()

        return jsonify([record.to_dict() for record in records])
    except Exception as e:
        current_app.logger.error(f"Get training history error: {str(e)}")
        return jsonify({"error": "Failed to get training history"}), 500

@history_bp.route('/training/<int:record_id>', methods=['GET'])
@login_required
def get_training_record(record_id):
    try:
        current_user = request.user
        record = TrainingRecord.query.filter_by(
            id=record_id,
            user_id=current_user["user_id"]
        ).first()

        if not record:
            return jsonify({"error": "Record not found"}), 404

        return jsonify(record.to_dict())
    except Exception as e:
        current_app.logger.error(f"Get training record error: {str(e)}")
        return jsonify({"error": "Failed to get training record"}), 500


@history_bp.route('/training/<int:record_id>', methods=['DELETE'])
@login_required
def delete_training_record(record_id):
    try:
        current_user = request.user
        record = TrainingRecord.query.filter_by(
            id=record_id,
            user_id=current_user["user_id"]
        ).first()

        if not record:
            return jsonify({"error": "Record not found"}), 404

        db.session.delete(record)
        db.session.commit()

        return jsonify({"message": "Record deleted successfully"})
    except Exception as e:
        current_app.logger.error(f"Delete training record error: {str(e)}")
        return jsonify({"error": "Failed to delete training record"}), 500


# 预测历史记录路由
@history_bp.route('/prediction', methods=['GET'])
@login_required
def get_prediction_history():
    try:
        current_user = request.user
        records = PredictRecord.query.filter_by(
            user_id=current_user["user_id"]
        ).order_by(PredictRecord.predict_time.desc()).all()

        return jsonify([record.to_dict() for record in records])
    except Exception as e:
        current_app.logger.error(f"Get prediction history error: {str(e)}")
        return jsonify({"error": "Failed to get prediction history"}), 500

@history_bp.route('/prediction/<int:record_id>', methods=['GET'])
@login_required
def get_prediction_record(record_id):
    try:
        current_user = request.user
        record = PredictRecord.query.filter_by(
            id=record_id,
            user_id=current_user["user_id"]
        ).first()

        if not record:
            return jsonify({"error": "Record not found"}), 404

        return jsonify(record.to_dict())
    except Exception as e:
        current_app.logger.error(f"Get prediction record error: {str(e)}")
        return jsonify({"error": "Failed to get prediction record"}), 500

@history_bp.route('/prediction/<int:record_id>', methods=['DELETE'])
@login_required
def delete_prediction_record(record_id):
    try:
        current_user = request.user
        record = PredictRecord.query.filter_by(
            id=record_id,
            user_id=current_user["user_id"]
        ).first()

        if not record:
            return jsonify({"error": "Record not found"}), 404

        db.session.delete(record)
        db.session.commit()

        return jsonify({"message": "Prediction record deleted successfully"})
    except Exception as e:
        current_app.logger.error(f"Delete prediction record error: {str(e)}")
        db.session.rollback()
        return jsonify({"error": "Failed to delete prediction record"}), 500

# 预处理历史记录路由
from app.models.file_model import UserFile  # 假设存在这个模型

@history_bp.route('/preprocessing', methods=['GET'])
@login_required
def get_preprocessing_history():
    try:
        current_user = request.user
        records = PreprocessingRecord.query.filter_by(
            user_id=current_user["user_id"]
        ).order_by(PreprocessingRecord.created_at.desc()).all()

        result = []
        for record in records:
            # 查询原始文件和处理后文件的详情
            original_file = UserFile.query.get(record.original_file_id)
            processed_file = UserFile.query.get(record.processed_file_id)
            processing_steps = PreprocessingStep.query.filter_by(preprocessing_record_id=record.id).all()

            result.append({
                'id': record.id,
                'user_id': record.user_id,
                'created_at': record.created_at,

                # 原始文件信息
                'original_file': {
                    'id': original_file.id,
                    'file_name': original_file.file_name,
                    'file_size': original_file.file_size,
                    'file_path': original_file.file_path,
                    'upload_time': original_file.upload_time.isoformat() if original_file else None
                } if original_file else None,

                # 处理后文件信息
                'processed_file': {
                    'id': processed_file.id,
                    'file_name': processed_file.file_name,
                    'file_size': processed_file.file_size,
                    'file_path': processed_file.file_path,
                    'upload_time': processed_file.upload_time.isoformat() if processed_file else None
                } if processed_file else None,

                # 其他字段可根据需要添加
                "processing_steps": [step.to_dict() for step in processing_steps]
            })

        return jsonify(result)
    except Exception as e:
        current_app.logger.error(f"Get preprocessing history error: {str(e)}")
        return jsonify({"error": "Failed to get preprocessing history"}), 500

@history_bp.route('/stacking_traing', methods=['GET'])
@login_required
def get_stacking_training_history():
    try:
        current_user = request.user
        # records = StackingTrainingRecord.query.filter_by(
        #     user_id=current_user["user_id"]
        # ).order_by(StackingTrainingRecord.start_time.desc()).all()
        records = StackingTrainingRecord.query.all()
        return jsonify([record.to_dict() for record in records])
    except Exception as e:
        current_app.logger.error(f"Get stacking training history error: {str(e)}")
        return jsonify({"error": "Failed to get stacking training history"}), 500

@history_bp.route('/stacking_prediction', methods=['GET'])
@login_required
def get_stacking_prediction_history():
    try:
        current_user = request.user
        records = StackingPredictionRecord.query.filter_by(
            user_id=current_user["user_id"]
        ).order_by(StackingPredictionRecord.start_time.desc()).all()
        return jsonify([record.to_dict() for record in records])
    except Exception as e:
        current_app.logger.error(f"Get stacking prediction history error: {str(e)}")