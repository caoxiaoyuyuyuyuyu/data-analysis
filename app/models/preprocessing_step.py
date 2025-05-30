from app.extensions import db
from datetime import datetime


class PreprocessingStep(db.Model):
    __tablename__ = 'preprocessing_steps'

    id = db.Column(db.Integer, primary_key=True)
    preprocessing_record_id = db.Column(db.Integer, db.ForeignKey('preprocessing_record.id', ondelete='CASCADE'), nullable=False)
    step_name = db.Column(db.String(100), nullable=False)
    step_order = db.Column(db.Integer, nullable=False)
    step_type = db.Column(db.String(100), nullable=False)
    parameters = db.Column(db.JSON, nullable=False, default={})
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    duration = db.Column(db.Float, nullable=False, default=0.0)

    def to_dict(self):
        return {
            'id': self.id,
            'preprocessing_record_id': self.preprocessing_record_id,
            'step_name': self.step_name,
            'step_order': self.step_order,
            'step_type': self.step_type,
            'parameters': self.parameters,
            'created_at': self.created_at,
            'duration': self.duration
        }
    def __repr__(self):
        return f'<PreprocessingStep {self.step_name} for preprocessing_record {self.preprocessing_record_id}>'