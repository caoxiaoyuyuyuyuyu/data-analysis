# preprocessing_history.py
from app.extensions import db
from datetime import datetime

class PreprocessingHistory(db.Model):
    __tablename__ = 'preprocessing_history'

    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey('user_files.id', ondelete='CASCADE'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    processed_filename = db.Column(db.String(255), nullable=False)
    processing_time = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    operation_type = db.Column(db.String(100), nullable=False)
    parameters = db.Column(db.JSON, nullable=False)
    duration = db.Column(db.Interval, nullable=False)
    rows_before = db.Column(db.Integer, nullable=False)
    rows_after = db.Column(db.Integer, nullable=False)
    columns_before = db.Column(db.Integer, nullable=False)
    columns_after = db.Column(db.Integer, nullable=False)

    # 修正关系定义
    user = db.relationship('User', backref=db.backref('preprocessing_history', lazy=True))
    file = db.relationship('UserFile', backref=db.backref('preprocessing_history', lazy=True))

    def __repr__(self):
        return f'<PreprocessingHistory {self.id} for file {self.file_id}>'