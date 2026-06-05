from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from config import Config, DevelopmentConfig
from models import db, User
from auth import AuthManager, token_required
import logging

app = Flask(__name__)
app.config.from_object(DevelopmentConfig)
CORS(app)
db.init_app(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with app.app_context():
    db.create_all()

@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if not all([username, email, password]):
            return jsonify({'error': 'Missing fields'}), 400
        
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username taken'}), 409
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        logger.info(f'User registered: {username}')
        return jsonify({'message': 'Success', 'user': user.to_dict()}), 201
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        db.session.rollback()
        return jsonify({'error': 'Failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        ip_address = request.remote_addr
        if AuthManager.check_login_attempts(user.id, ip_address):
            return jsonify({'error': 'Too many attempts'}), 429
        
        user.last_login = datetime.utcnow()
        access_token = AuthManager.generate_access_token(user.id, user.role)
        refresh_token = AuthManager.create_session(user.id, ip_address)
        AuthManager.log_login_attempt(user.id, ip_address, success=True)
        
        db.session.commit()
        logger.info(f'User logged in: {username}')
        
        return jsonify({
            'access_token': access_token,
            'refresh_token': refresh_token,
            'user': user.to_dict()
        }), 200
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        return jsonify({'error': 'Failed'}), 500

@app.route('/api/auth/refresh', methods=['POST'])
def refresh():
    """Refresh access token"""
    try:
        data = request.get_json()
        refresh_token = data.get('refresh_token')
        
        if not refresh_token:
            return jsonify({'error': 'Token required'}), 400
        
        payload = AuthManager.verify_token(refresh_token, 'refresh')
        if not payload:
            return jsonify({'error': 'Invalid token'}), 401
        
        user = User.query.get(payload['user_id'])
        if not user or not user.is_active:
            return jsonify({'error': 'User not found'}), 401
        
        new_access_token = AuthManager.generate_access_token(user.id, user.role)
        return jsonify({'access_token': new_access_token}), 200
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        return jsonify({'error': 'Failed'}), 500

@app.route('/api/auth/logout', methods=['POST'])
@token_required
def logout():
    """User logout"""
    try:
        data = request.get_json()
        refresh_token = data.get('refresh_token')
        
        if refresh_token:
            AuthManager.revoke_session(refresh_token)
        
        logger.info(f'User logged out: {request.user_id}')
        return jsonify({'message': 'Logout successful'}), 200
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        return jsonify({'error': 'Failed'}), 500

@app.route('/api/users/profile', methods=['GET'])
@token_required
def get_profile():
    """Get user profile"""
    try:
        user = User.query.get(request.user_id)
        if not user:
            return jsonify({'error': 'Not found'}), 404
        return jsonify(user.to_dict()), 200
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        return jsonify({'error': 'Failed'}), 500

@app.route('/api/users/profile', methods=['PUT'])
@token_required
def update_profile():
    """Update user profile"""
    try:
        user = User.query.get(request.user_id)
        if not user:
            return jsonify({'error': 'Not found'}), 404
        
        data = request.get_json()
        if 'first_name' in data:
            user.first_name = data['first_name']
        if 'last_name' in data:
            user.last_name = data['last_name']
        
        user.updated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f'Profile updated: {request.user_id}')
        return jsonify(user.to_dict()), 200
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        db.session.rollback()
        return jsonify({'error': 'Failed'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'service': 'auth'}), 200

if __name__ == '__main__':
    logger.info('Starting Auth System...')
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000)
