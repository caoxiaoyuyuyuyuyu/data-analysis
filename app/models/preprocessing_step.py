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

    def __repr__(self):
        return f'<PreprocessingStep {self.step_name} for preprocessing_record {self.preprocessing_record_id}>'