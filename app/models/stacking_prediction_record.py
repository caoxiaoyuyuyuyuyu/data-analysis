from app.extensions import db
from datetime import datetime

class StackingPredictionRecord(db.Model):
    __tablename__ = 'stacking_prediction_records'

    id = db.Column(db.UUID, primary_key=True, default=db.func.gen_random_uuid())
    training_record_id = db.Column(
        db.Integer,
        db.ForeignKey('stacking_training_records.id'),
        nullable=False
    )
    input_file_id = db.Column(
        db.Integer,
        db.ForeignKey('user_files.id'),
        nullable=False
    )
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id'),
        nullable=False
    )
    start_time = db.Column(db.DateTime(timezone=True), nullable=False)
    end_time = db.Column(db.DateTime(timezone=True), nullable=False)
    result_path = db.Column(db.Text)
    result_summary = db.Column(db.JSON)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    updated_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())

    # 关系定义
    file = db.relationship('UserFile', back_populates='stacking_prediction_records')
    stacking_training_record = db.relationship(
        "StackingTrainingRecord",
        back_populates="stacking_prediction_records"
    )