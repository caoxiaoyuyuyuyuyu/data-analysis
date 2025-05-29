# training_record.py
from app.extensions import db
from datetime import datetime

class TrainingRecord(db.Model):
    __tablename__ = 'training_records'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'))
    file_id = db.Column(db.Integer, db.ForeignKey('user_files.id', ondelete='CASCADE'), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    model_name = db.Column(db.String(100), nullable=False)
    model_type = db.Column(db.String(100), nullable=False)
    test_size = db.Column(db.Float, nullable=False)
    target_column = db.Column(db.String(100), nullable=False)
    duration = db.Column(db.Float, nullable=False)  # 改为Float对应DOUBLE PRECISION
    metrics = db.Column(db.JSON, nullable=False)  # 对应JSONB
    model_parameters = db.Column(db.JSON, nullable=False)  # 对应JSONB
    learning_curve = db.Column(db.JSON, nullable=False)  # 新增字段
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)  # 新增字段
    updated_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)  # 新增字段
    model_file_path = db.Column(db.String(512), nullable=False)  # 原model_path重命名
    model_file_size = db.Column(db.BigInteger, nullable=False)  # 新增字段

    # 关系定义
    user = db.relationship('User', back_populates='training_records')
    file = db.relationship('UserFile', back_populates='training_records')

    # 添加 to_dict 方法
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'file_id': self.file_id,
            'file_name': self.file_name,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'test_size': self.test_size,
            'target_column': self.target_column,
            'duration': self.duration,
            'metrics': self.metrics,
            'model_parameters': self.model_parameters,
            'learning_curve': self.learning_curve,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'model_file_path': self.model_file_path,
            'model_file_size': self.model_file_size
        }
    def __repr__(self):
        return f'<TrainingRecord {self.id} - {self.model_name}>'