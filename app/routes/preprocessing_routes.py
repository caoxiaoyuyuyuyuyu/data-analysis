# preprocessing_routes.py
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request, current_app
from app.models.file_model import UserFile
from app.models.preprocessing_history import PreprocessingHistory
from app.models.preprocessing_step import PreprocessingStep
from app.utils.data_processor import DataProcessor
from app.utils.jwt_utils import login_required
from app.extensions import db
import pandas as pd

preprocessing_bp = Blueprint('preprocessing', __name__, url_prefix='/preprocessing')
@preprocessing_bp.route('/<int:file_id>/data', methods=['GET'])
@login_required
def get_preprocessing_data(file_id):
    try:
        current_user = request.user
        file_record = UserFile.query.filter_by(
            id=file_id,
            user_id=current_user["user_id"]
        ).first()

        if not file_record:
            return jsonify({"error": "File not found"}), 404

        # 加载数据
        df = load_dataframe(file_record)

        if not isinstance(df, pd.DataFrame):
            return df

        # 获取数据预览
        preview = get_data_preview(df)

        # 获取统计信息
        stats = DataProcessor.get_data_summary(df)

        return jsonify({
            "metadata": {
                "file_id": file_record.id,
                "file_name": file_record.file_name,
                "rows": df.shape[0],
                "columns": df.shape[1]
            },
            "preview": preview,
            "statistics": stats
        })

    except Exception as e:
        current_app.logger.error(f"Data processing error: {str(e)}")
        return jsonify({"error": "Data processing failed"}), 500

from app.core.data_loader import dataloader
def load_dataframe(file_record):
    """加载数据文件到DataFrame"""
    try:
        return dataloader.load_file(file_record.file_path)
    except Exception as e:
        current_app.logger.error(f"File loading error: {str(e)}")
        return jsonify({"error": f"Failed to load file: {str(e)}"}), 500


def get_data_preview(df, rows=10):
    """获取数据预览"""
    return {
        "columns": list(df.columns),
        "sample_data": df.head(rows).fillna('').to_dict(orient='records')
    }

@preprocessing_bp.route('/<int:file_id>', methods=['POST'])
@login_required
def apply_preprocessing_step(file_id):
    try:
        start_time = datetime.utcnow()
        current_user = request.user
        data = request.get_json()

        # Validate request
        if not data or 'step' not in data:
            return jsonify({"error": "Missing step data"}), 400

        step_data = data['step']

        # Verify file exists and belongs to user
        file_record = UserFile.query.filter_by(
            id=file_id,
            user_id=current_user["user_id"]
        ).first()

        if not file_record:
            return jsonify({"error": "File not found"}), 404

        # Load the dataframe
        df = load_dataframe(file_record)
        if not isinstance(df, pd.DataFrame):
            return df

        # Record stats before processing
        rows_before = df.shape[0]
        cols_before = df.shape[1]

        # Process the step based on type
        processor = DataProcessor()
        step_type = step_data['type']
        params = step_data.get('params', {})

        try:
            if step_type == 'missing_values':
                processed_df = processor.handle_missing_values(
                    df,
                    strategy=params.get('strategy', 'mean'),
                    fill_value=params.get('fill_value'),
                    columns=params.get('columns')
                )
            elif step_type == 'feature_scaling':
                processed_df = processor.apply_feature_scaling(
                    df,
                    method=params.get('method'),
                    columns=params.get('columns')
                )
            elif step_type == 'encoding':
                processed_df = processor.encode_categorical(
                    df,
                    method=params.get('method'),
                    columns=params.get('columns')
                )
            else:
                return jsonify({"error": "Invalid step type"}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Record stats after processing
        rows_after = processed_df.shape[0]
        cols_after = processed_df.shape[1]

        # Save the processed data back to file
        original_filename = file_record.file_name
        processed_filename = save_processed_data(file_record, processed_df)  # 修改此行


        # Create preprocessing step record
        new_step = PreprocessingStep(
            file_id=file_id,
            step_name=step_type,
            step_order=get_next_step_order(file_id),
            parameters=params
        )
        db.session.add(new_step)

        # Create history record
        history = PreprocessingHistory(
            file_id=file_id,
            user_id=current_user["user_id"],
            original_filename=original_filename,
            processed_filename=processed_filename,
            operation_type=step_type,
            parameters=params,
            duration=datetime.utcnow() - start_time,
            rows_before=rows_before,
            rows_after=rows_after,
            columns_before=cols_before,
            columns_after=cols_after
        )
        db.session.add(history)

        db.session.commit()

        return jsonify({
            "message": f"{step_type} applied successfully",
            "history_id": history.id,
            "stats": {
                "rows_before": rows_before,
                "rows_after": rows_after,
                "columns_before": cols_before,
                "columns_after": cols_after
            }
        })

    except Exception as e:
        current_app.logger.error(f"Preprocessing error: {str(e)}")
        db.session.rollback()
        return jsonify({
            "error": "Preprocessing failed",
            "details": str(e)
        }), 500
def get_next_step_order(file_id):
    """Get the next step order number for a file"""
    last_step = PreprocessingStep.query.filter_by(file_id=file_id) \
        .order_by(PreprocessingStep.step_order.desc()).first()
    return last_step.step_order + 1 if last_step else 1

from pathlib import Path
import os


def save_processed_data(file_record, df):
    try:
        original_path = Path(file_record.file_path)
        processed_dir = original_path.parent / "processed"
        os.makedirs(processed_dir, exist_ok=True)  # 创建processed目录

        # 生成唯一的新文件名（例如：原文件名_时间戳.扩展名）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_filename = f"{original_path.stem}_{timestamp}{original_path.suffix}"
        processed_path = processed_dir / processed_filename

        # 保存到新路径
        if file_record.file_type == 'csv':
            df.to_csv(processed_path, index=False)
        elif file_record.file_type in ['xlsx', 'xls']:
            df.to_excel(processed_path, index=False)
        elif file_record.file_type == 'json':
            df.to_json(processed_path, orient='records')
        # ...其他格式

        # 更新文件记录，指向新文件
        file_record.file_path = str(processed_path)
        file_record.file_name = processed_filename

        return processed_filename  # 返回实际生成的文件名

    except Exception as e:
        current_app.logger.error(f"Failed to save processed data: {str(e)}")
        raise