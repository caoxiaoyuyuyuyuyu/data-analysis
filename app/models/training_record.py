# training_record.py
from app.extensions import db
from datetime import datetime

class TrainingRecord(db.Model):
    __tablename__ = 'training_records'

    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey('user_files.id', ondelete='CASCADE'))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'))
    file_name = db.Column(db.String(255))
    model_name = db.Column(db.String(100), nullable=False)
    model_parameters = db.Column(db.JSON, nullable=False)
    training_time = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    metrics = db.Column(db.JSON, nullable=False)
    model_path = db.Column(db.String(512), nullable=False)
    feature_importance = db.Column(db.JSON)
    duration = db.Column(db.Interval, nullable=False)  # 改为 Float 类型，单位为秒
    is_optimized = db.Column(db.Boolean, default=False)

    user = db.relationship('User', backref='training_records')
    file = db.relationship('UserFile', backref='training_records')