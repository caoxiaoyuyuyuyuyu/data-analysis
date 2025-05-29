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
    status = db.Column(db.String(20), nullable=False, server_default='pending')
    error_message = db.Column(db.Text)

    # 关系定义
    user = db.relationship('User', backref='predict_records')
    training_record = db.relationship('TrainingRecord', backref='predictions')

    def __init__(self, user_id, training_record_id=None,
                 input_file_id=None, input_data=None, parameters=None):
        self.user_id = user_id
        self.training_record_id = training_record_id
        self.input_file_id = input_file_id
        self.input_data = input_data
        self.parameters = parameters
        self.status = PredictStatus.PENDING.value
        self.predict_duration = 0.0

    def update_result(self, output_data, duration, status=PredictStatus.COMPLETED, error_message=None):
        self.output_data = output_data
        self.predict_duration = duration
        self.status = status.value
        self.error_message = error_message
        self.predict_time = datetime.utcnow()

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'training_record_id': self.training_record_id,
            'input_file_id': self.input_file_id,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'predict_time': self.predict_time.isoformat() if self.predict_time else None,
            'predict_duration': self.predict_duration,
            'parameters': self.parameters,
            'status': self.status,
            'error_message': self.error_message
        }

    @classmethod
    def create_from_training(cls, training_record, input_data=None, input_file_id=None, parameters=None):
        return cls(
            user_id=training_record.user_id,
            training_record_id=training_record.id,
            input_file_id=input_file_id,
            input_data=input_data,
            parameters=parameters
        )