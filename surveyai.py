from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import logging
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise ValueError("GROQ_API_KEY environment variable is not set")

# Initialize Flask app
app = Flask(__name__)

def make_groq_request(prompt):
    """Make a request to Groq API"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in Groq request: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_completion():
    try:
        data = request.json
        logger.debug(f"Received request data: {data}")
        
        prompt = data.get('prompt')
        if not prompt:
            logger.warning("No prompt provided in request")
            return jsonify({'error': 'No prompt provided'}), 400

        try:
            response = make_groq_request(prompt)
            return jsonify({'response': response})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    except Exception as e:
        logger.error(f"Error in run_completion: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 8000))
    
    print(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=True)