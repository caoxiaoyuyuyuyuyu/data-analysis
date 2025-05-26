from app.extensions import db

class UserFile(db.Model):
    __tablename__ = 'user_files'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'))
    file_name = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    file_size = db.Column(db.BigInteger, nullable=False)
    file_type = db.Column(db.String(50))
    upload_time = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    description = db.Column(db.Text)
    is_processed = db.Column(db.Boolean, default=False)

    user = db.relationship('User', backref='files')