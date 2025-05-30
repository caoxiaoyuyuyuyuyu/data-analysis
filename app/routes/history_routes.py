# history_routes.py
from flask import Blueprint, jsonify, request, current_app
from app.models.training_record import TrainingRecord
from app.models.prediction_record import PredictionRecord
from app.models.preprocessing_history import PreprocessingHistory
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
        records = PredictionRecord.query.filter_by(
            user_id=current_user["user_id"]
        ).order_by(PredictionRecord.prediction_time.desc()).all()

        return jsonify([{
            "id": record.id,
            "model_id": record.model_id,
            "model_name": record.model_name,
            "prediction_time": record.prediction_time.isoformat(),
            "input_summary": record.input_summary,
            "output_summary": record.output_summary
        } for record in records])
    except Exception as e:
        current_app.logger.error(f"Get prediction history error: {str(e)}")
        return jsonify({"error": "Failed to get prediction history"}), 500


# 预处理历史记录路由
@history_bp.route('/preprocessing', methods=['GET'])
@login_required
def get_preprocessing_history():
    try:
        current_user = request.user
        records = PreprocessingHistory.query.filter_by(
            user_id=current_user["user_id"]
        ).order_by(PreprocessingHistory.processing_time.desc()).all()

        return jsonify([{
            "id": record.id,
            "file_id": record.file_id,
            "original_filename": record.original_filename,
            "processed_filename": record.processed_filename,
            "processing_time": record.processing_time.isoformat(),
            "operation_type": record.operation_type,
            "parameters": record.parameters,
            "duration": record.duration.total_seconds(),
            "rows_before": record.rows_before,
            "rows_after": record.rows_after,
            "columns_before": record.columns_before,
            "columns_after": record.columns_after
        } for record in records])
    except Exception as e:
        current_app.logger.error(f"Get preprocessing history error: {str(e)}")
        return jsonify({"error": "Failed to get preprocessing history"}), 500