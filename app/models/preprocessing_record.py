# preprocessing_record.py
from app.extensions import db
from datetime import datetime

class PreprocessingRecord(db.Model):
    __tablename__ = 'preprocessing_record'

    id = db.Column(db.Integer, primary_key=True)
    original_file_id = db.Column(db.Integer, db.ForeignKey('user_files.id', ondelete='CASCADE'), nullable=False)
    processed_file_id = db.Column(db.Integer, db.ForeignKey('user_files.id', ondelete='CASCADE'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    rows_ori = db.Column(db.Integer, nullable=False)
    rows_current = db.Column(db.Integer, nullable=False)
    columns_ori = db.Column(db.Integer, nullable=False)
    columns_current = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<PreprocessingRecord {self.id} for file {self.original_file_id}>'