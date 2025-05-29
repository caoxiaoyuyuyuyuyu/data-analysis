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
    parent_id = db.Column(db.Integer, db.ForeignKey('user_files.id'), default=0)  # 新增parent_id

    # 关系定义
    user = db.relationship('User', back_populates='files')
    training_records = db.relationship("TrainingRecord", back_populates="file", cascade="all, delete-orphan")

    # 自引用关系
    parent = db.relationship('UserFile', remote_side=[id], backref='children', post_update=True)