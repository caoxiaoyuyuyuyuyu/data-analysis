# app/models/model_config.py
from app.extensions import db
from datetime import datetime


class ModelConfig(db.Model):
    __tablename__ = 'model_configs'

    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(100), unique=True, nullable=False)
    display_name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(20), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    parameters = db.relationship('ModelParameter', backref='config', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<ModelConfig {self.model_type}>'


class ModelParameter(db.Model):
    __tablename__ = 'model_parameters'

    id = db.Column(db.Integer, primary_key=True)
    model_config_id = db.Column(db.Integer, db.ForeignKey('model_configs.id', ondelete='CASCADE'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(20), nullable=False)
    default_value = db.Column(db.JSON)
    min_value = db.Column(db.Numeric)
    max_value = db.Column(db.Numeric)
    step_size = db.Column(db.Numeric)
    options = db.Column(db.JSON)
    description = db.Column(db.Text)
    param_order = db.Column(db.Integer, default=0)
    is_required = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('model_config_id', 'name', name='uq_model_parameter'),
    )

    def __repr__(self):
        return f'<ModelParameter {self.name} for {self.model_config_id}>'