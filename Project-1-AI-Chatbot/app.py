from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import logging
from config import Config
from database import ChatbotDatabase
from chatbot import ChatbotNLP

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

db = ChatbotDatabase()
chatbot = ChatbotNLP()

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Serve chatbot web interface"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); overflow: hidden; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }
            .chat-box { height: 400px; overflow-y: auto; padding: 20px; background: #f8f9fa; border-bottom: 1px solid #ddd; }
            .message { margin: 10px 0; display: flex; }
            .user-msg { justify-content: flex-end; }
            .user-msg p { background: #007bff; color: white; padding: 10px 15px; border-radius: 15px; max-width: 70%; }
            .bot-msg { justify-content: flex-start; }
            .bot-msg p { background: white; color: #333; padding: 10px 15px; border-radius: 15px; max-width: 70%; border: 1px solid #ddd; }
            .input-area { padding: 15px; display: flex; gap: 10px; }
            input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 20px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 20px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 AI Chatbot</h1>
                <p>How can I help you today?</p>
            </div>
            <div class="chat-box" id="chatBox"></div>
            <div class="input-area">
                <input type="text" id="userInput" placeholder="Type your message..." onkeypress="if(event.key=='Enter') sendMessage()">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        <script>
            function sendMessage() {
                const input = document.getElementById('userInput').value.trim();
                if (!input) return;
                
                addMessage('user', input);
                document.getElementById('userInput').value = '';
                
                fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: input,
                        user_id: localStorage.getItem('userId') || (localStorage.setItem('userId', 'user_' + Math.random().toString(36).substr(2, 9)), localStorage.getItem('userId'))
                    })
                })
                .then(r => r.json())
                .then(data => addMessage('bot', data.response))
                .catch(e => addMessage('bot', 'Error: ' + e.message));
            }
            
            function addMessage(sender, text) {
                const box = document.getElementById('chatBox');
                const div = document.createElement('div');
                div.className = 'message ' + (sender === 'user' ? 'user-msg' : 'bot-msg');
                const p = document.createElement('p');
                p.textContent = text;
                div.appendChild(p);
                box.appendChild(div);
                box.scrollTop = box.scrollHeight;
            }
        </script>
    </body>
    </html>
    '''

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        user_id = data.get('user_id', str(uuid.uuid4()))
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        db.add_user(user_id)
        intent, confidence = chatbot.detect_intent(user_message)
        response, detected_intent, intent_confidence = chatbot.generate_response(user_message, intent, confidence)
        sentiment = chatbot.analyze_sentiment(user_message)
        
        db.log_conversation(user_id, session_id, user_message, response, detected_intent, intent_confidence)
        logger.info(f"Chat: {user_message[:50]}...")
        
        return jsonify({
            'response': response,
            'user_id': user_id,
            'session_id': session_id,
            'intent': detected_intent,
            'confidence': float(intent_confidence),
            'sentiment': sentiment
        })
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs/<user_id>', methods=['GET'])
def get_logs(user_id):
    """Get conversation logs"""
    try:
        limit = request.args.get('limit', 50, type=int)
        logs = db.get_conversation_history(user_id, limit)
        return jsonify({'logs': [dict(log) for log in logs]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/faq', methods=['POST'])
def add_faq():
    """Add FAQ"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        answer = data.get('answer', '').strip()
        keywords = data.get('keywords', '')
        
        if not question or not answer:
            return jsonify({'error': 'Missing fields'}), 400
        
        db.add_faq(question, answer, keywords)
        return jsonify({'message': 'FAQ added'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/faqs', methods=['GET'])
def get_faqs():
    """Get all FAQs"""
    try:
        faqs = db.get_all_faqs()
        return jsonify({'faqs': [dict(faq) for faq in faqs]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'service': 'chatbot'})

if __name__ == '__main__':
    logger.info("Starting AI Chatbot...")
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000)
