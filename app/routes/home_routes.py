from flask import Blueprint

home_bp = Blueprint('home', __name__)

@home_bp.route('/')
def hello_world():
    return 'Hello World!'