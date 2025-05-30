from datetime import datetime
from enum import Enum
from sqlalchemy.dialects.postgresql import JSONB
from app.extensions import db

class PredictStatus(Enum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'

class PredictRecord(db.Model):
    __tablename__ = 'predict_records'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id',  ondelete='CASCADE'), nullable=False)
    training_record_id = db.Column(db.Integer, db.ForeignKey('training_records.id', ondelete='CASCADE'))
    input_file_id = db.Column(db.Integer, db.ForeignKey('user_files.id', ondelete='CASCADE'))
    output_file_path = db.Column(db.String(100))
    predict_time = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    predict_duration = db.Column(db.Float, nullable=False)
    parameters = db.Column(JSONB)
    status = db.Column(db.String(200), nullable=False, server_default='pending')
    error_message = db.Column(db.Text)

    # 关系定义
    user = db.relationship('User', backref='predict_records')
    training_record = db.relationship('TrainingRecord', backref='predictions')

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'training_record_id': self.training_record_id,
            'input_file_id': self.input_file_id,
            'output_file_path': self.output_file_path,
            'predict_time': self.predict_time,
            'predict_duration': self.predict_duration,
            'parameters': self.parameters,
            'status': self.status,
            'error_message': self.error_message
        }