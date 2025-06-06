# preprocessing_routes.py
import uuid
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request, current_app, session
from app.models.file_model import UserFile
from app.models.preprocessing_record import PreprocessingRecord
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

def process_data_step(df, step_type, params, processor):
    """根据步骤类型应用预处理步骤"""
    if step_type == 'missing_values':
        return processor.handle_missing_values(
            df,
            strategy=params.get('strategy', 'mean'),
            fill_value=params.get('fill_value'),
            columns=params.get('columns')
        )
    elif step_type == 'feature_scaling':
        # 特性	标准化 (Standardization / Z-Score Normalization)	归一化 (Normalization / Min-Max Scaling)	鲁棒缩放 (Robust Scaling)
        # 核心思想	使数据服从均值为 0，标准差为 1的标准正态分布。	将数据线性变换到指定的范围（通常是 [0, 1]）。	使用中位数和四分位数范围缩放，抵抗异常值。
        # 验证缩放参数
        method = params.get('method', 'standard')
        columns = params.get('columns')

        if not columns:
            raise ValueError("Missing 'columns' parameter for scaling")

        if method not in ['standard', 'minmax', 'robust']:
            raise ValueError(f"Invalid scaling method: {method}. Must be 'standard', 'minmax' or 'robust'")

        # 执行缩放
        return processor.apply_feature_scaling(df, method, columns)
    elif step_type == 'encoding':
        return processor.encode_categorical(
            df,
            method=params.get('method'),
            columns=params.get('columns')
        )
    elif step_type == 'pca':
        n_components = params.get('n_components', 2)
        return processor.apply_pca(
            df,
            n_components=n_components,
            columns=params.get('columns',  None)
        )
    elif step_type == 'outlier_handling':
        method = params.get('method', 'zscore')
        threshold = params.get('threshold', 3)
        return processor.handle_outliers(
            df,
            method=method,
            threshold=threshold,
            columns=params.get('columns')
        )
    elif step_type == 'feature_selection':
        return processor.select_features(
            df,
            columns=params.get('columns')
        )
    else:
        raise ValueError(f"不支持的步骤类型: {step_type}")

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
        step_type = step_data['type']
        params = step_data.get('params', {})
        process_record_id = data['processed_record_id']

        if process_record_id:
            process_record = PreprocessingRecord.query.filter_by(id=process_record_id).first()
            ori_file_id = process_record.original_file_id
            preprocessed_file_id = process_record.processed_file_id
            # Verify file exists and belongs to user
            file_record = UserFile.query.filter_by(
                id=preprocessed_file_id,
                user_id=current_user["user_id"]
            ).first()
        else:
            ori_file_id = file_id
            process_record = PreprocessingRecord(
                user_id=current_user["user_id"],
                original_file_id=file_id
            )
            preprocessed_file_id = None
            file_record = UserFile.query.filter_by(
                id=ori_file_id,
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
        processor = DataProcessor()

        # 应用预处理步骤
        try:
            processed_result = process_data_step(df, step_type, params, processor)
            current_app.logger.info(f"processed_result: {processed_result}")
            processed_df = processed_result['processed_df']
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Record stats after processing
        rows_after = processed_df.shape[0]
        cols_after = processed_df.shape[1]

        current_app.logger.info(f"preprocessed_file_id: {preprocessed_file_id}")
        # Save the processed data back to file
        processed_file = save_processed_data(db, file_record, processed_df, preprocessed_file_id)  # 修改此行
        processed_file_id = processed_file.id
        #     rows_ori = db.Column(db.Integer, nullable=False)
        #     rows_current = db.Column(db.Integer, nullable=False)
        #     columns_ori = db.Column(db.Integer, nullable=False)
        #     columns_current = db.Column(db.Integer, nullable=False)
        process_record.rows_ori = rows_before
        process_record.rows_current = rows_after
        process_record.columns_ori = cols_before
        process_record.columns_current = cols_after
        process_record.processed_file_id = processed_file_id
        db.session.add(process_record)
        db.session.flush()  # 获取生成的ID

        processed_record_id = process_record.id

        steps = PreprocessingStep.query.filter_by(preprocessing_record_id=processed_record_id).all()

        new_step = PreprocessingStep(
            preprocessing_record_id=processed_record_id,
            step_name=step_type,
            step_order=len(steps) + 1,
            step_type=step_type,
            parameters=params,
            duration= (datetime.utcnow() - start_time).total_seconds()
        )
        db.session.add(new_step)

        db.session.commit()
        return jsonify({
            "processed_record_id": processed_record_id,
            "processed_file_id": processed_file_id,
            "details": processed_result.pop("processed_df")
        })
    except Exception as e:
        current_app.logger.error(f"Preprocessing error: {str(e)}")
        db.session.rollback()
        return jsonify({
            "error": "Preprocessing failed",
            "details": str(e)
        }), 500

    #

    #     # Create preprocessing step record
    #     new_step = PreprocessingStep(
    #         file_id=file_id,
    #         step_name=step_type,
    #         step_order=get_next_step_order(file_id),
    #         parameters=params
    #     )
    #     db.session.add(new_step)
    #
    #     # Create history record
    #     history = PreprocessingHistory(
    #         file_id=file_id,
    #         user_id=current_user["user_id"],
    #         original_filename=original_filename,
    #         processed_filename=processed_filename,
    #         operation_type=step_type,
    #         parameters=params,
    #         duration=datetime.utcnow() - start_time,
    #         rows_before=rows_before,
    #         rows_after=rows_after,
    #         columns_before=cols_before,
    #         columns_after=cols_after
    #     )
    #     db.session.add(history)
    #
    #     db.session.commit()
    #
    #     return jsonify({
    #         "message": f"{step_type} applied successfully",
    #         "history_id": history.id,
    #         "stats": {
    #             "rows_before": rows_before,
    #             "rows_after": rows_after,
    #             "columns_before": cols_before,
    #             "columns_after": cols_after
    #         }
    #     })
    #
def get_next_step_order(file_id):
    """Get the next step order number for a file"""
    last_step = PreprocessingStep.query.filter_by(file_id=file_id) \
        .order_by(PreprocessingStep.step_order.desc()).first()
    return last_step.step_order + 1 if last_step else 1

from pathlib import Path
import os


def save_processed_data(db, file_record, df, preprocessed_file_id):
    try:
        if preprocessed_file_id:
            processed_file = UserFile.query.filter_by(id=preprocessed_file_id).first()
            if processed_file:
                processed_path = processed_file.file_path
                current_app.logger.info(f"processed_path: {processed_path}")
                ori_df =  dataloader.load_file(processed_path)
                current_app.logger.info(f"ori_df: {ori_df.columns}")
                current_app.logger.info(f"df: {df.columns}")
                # 保存到新路径
                if file_record.file_type == 'csv':
                    df.to_csv(processed_path, index=False)
                elif file_record.file_type in ['xlsx', 'xls']:
                    df.to_excel(processed_path, index=False)
                elif file_record.file_type == 'json':
                    df.to_json(processed_path, orient='records')
                processed_file.file_size = os.path.getsize(processed_path)
                db.session.add(processed_file)
                db.session.flush()  # 获取生成的ID
                return processed_file

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

        new_file=UserFile(
            user_id=file_record.user_id,
            file_name=processed_filename,
            file_path=str(processed_path),
            file_size=os.path.getsize(processed_path),
            file_type=file_record.file_type,
            parent_id=file_record.id,
        )
        db.session.add(new_file)
        db.session.flush()  # 获取生成的ID

        current_app.logger.info(f"Saved processed data to {new_file.id}")

        return new_file

    except Exception as e:
        db.session.rollback()
        # os.remove(processed_path)
        current_app.logger.error(f"Failed to save processed data: {str(e)}")
        raise