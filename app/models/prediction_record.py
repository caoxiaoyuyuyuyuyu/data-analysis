# prediction_record.py
from app.extensions import db
from datetime import datetime

class PredictionRecord(db.Model):
    __tablename__ = 'prediction_records'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'))
    model_id = db.Column(db.Integer, db.ForeignKey('training_records.id', ondelete='SET NULL'))
    input_data = db.Column(db.JSON)
    output_data = db.Column(db.JSON)
    prediction_time = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    input_file_id = db.Column(db.Integer, db.ForeignKey('user_files.id', ondelete='SET NULL'))
    input_summary = db.Column(db.Text)
    output_summary = db.Column(db.Text)
    model_name = db.Column(db.String(100))

    user = db.relationship('User', backref='prediction_records')
    model = db.relationship('TrainingRecord', backref='predictions')
    input_file = db.relationship('UserFile', backref='predictions')
