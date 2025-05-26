from app.extensions import db
from datetime import datetime


class PreprocessingStep(db.Model):
    __tablename__ = 'preprocessing_steps'

    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey('user_files.id', ondelete='CASCADE'), nullable=False)
    step_name = db.Column(db.String(100), nullable=False)
    step_order = db.Column(db.Integer, nullable=False)
    parameters = db.Column(db.JSON, nullable=False, default={})
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<PreprocessingStep {self.step_name} for file {self.file_id}>'