from app.extensions import db
from datetime import datetime

class StackingTrainingRecord(db.Model):
    __tablename__ = 'stacking_training_records'

    id = db.Column(db.Integer, primary_key=True)
    input_file_id = db.Column(db.Integer, db.ForeignKey('user_files.id', ondelete='CASCADE'), nullable=False)
    task_type = db.Column(db.String(50), nullable=False)
    base_model_names = db.Column(db.ARRAY(db.String(255)), nullable=False)
    meta_model_name = db.Column(db.String(255), nullable=False)
    cross_validation = db.Column(db.Integer, nullable=False)
    target = db.Column(db.String(255), nullable=False)
    model_id = db.Column(db.String(255), nullable=False)
    model_path = db.Column(db.Text, nullable=False)
    start_time = db.Column(db.DateTime(timezone=True), nullable=False)
    end_time = db.Column(db.DateTime(timezone=True), nullable=False)
    metrics = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    updated_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())

    # 关系定义
    file = db.relationship('UserFile', back_populates='stacking_training_records')
    stacking_prediction_records = db.relationship(
        "StackingPredictionRecord",
        back_populates="stacking_training_record",
        cascade="all, delete-orphan"
    )