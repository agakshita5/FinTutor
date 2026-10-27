import os
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

from finance_ai_chatbot import FinanceAIChatbot

load_dotenv()

app = Flask(__name__)

MAX_MESSAGE_CHARS = 1000

try:
    bot = FinanceAIChatbot(os.getenv("GROQ_API_KEY"))
    print(f"Chatbot initialized successfully using model: {bot.model_name}")
except Exception as e:
    print(f"Error initializing chatbot: {e}")
    bot = None


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/chat', methods=['POST'])
def chat():
    if not bot:
        return jsonify({'response': 'Chatbot not initialized properly. Check the server logs.',
                        'error': True}), 503

    message = (request.get_json(silent=True) or {}).get('message', '').strip()
    if not message:
        return jsonify({'response': 'Please enter a message.', 'error': False})
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify({'response': f'Please keep questions under {MAX_MESSAGE_CHARS} characters.',
                        'error': True}), 400

    r = bot.get_response(message)
    return jsonify({
        'response': r['answer'],
        'status': r['status'],
        'sources': r['sources'],
        'cached': r['cached'],
        'latency_ms': r['latency_ms'],
        'error': r['status'] == 'error',
    }), 502 if r['status'] == 'error' else 200


@app.route('/stats', methods=['GET'])
def stats():
    if not bot:
        return jsonify({'error': 'Chatbot not initialized'}), 503
    return jsonify({
        'total_queries': bot.total_queries,
        'cache_entries': len(bot.response_cache),
        'uptime_minutes': (datetime.now() - bot.session_start).seconds // 60,
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'chatbot_ready': bot is not None})


if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0')
