from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import logging
from groq import Groq  
from llm_agent import make_groq_request
import PyPDF2
from data_chunk import extract_text_from_pdf, chunk_text, create_embeddings, retrieve_relevant_chunks
from openai import OpenAI
from werkzeug.utils import secure_filename
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
logger.debug(f"Loading .env from: {env_path}")
load_dotenv(env_path)
logger.debug(f"GROQ_API_KEY exists: {bool(os.getenv('GROQ_API_KEY'))}")
logger.debug(f"OPENAI_API_KEY exists: {bool(os.getenv('OPENAI_API_KEY'))}")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key= OPENAI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise ValueError("GROQ_API_KEY environment variable is not set")

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_pdf(filepath):
    """Check if the file is a valid PDF by trying to read it"""
    try:
        with open(filepath, 'rb') as file:
            PyPDF2.PdfReader(file)
        return True
    except Exception:
        return False

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/questions')
def questions():
    return render_template('questions.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['pdf']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Verify it's a valid PDF
            if not is_valid_pdf(filepath):
                os.remove(filepath)
                return jsonify({'error': 'File is not a valid PDF'}), 400
            
            # Store the PDF text in session (in a real app, you'd use a proper session/store)
            app.config['CURRENT_PDF_TEXT'] = extract_text_from_pdf(filepath)
            
            return jsonify({'message': 'File uploaded successfully'}), 200
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        logger.debug(f"Received question data: {data}")
        
        question = data.get('question')
        if not question:
            logger.warning("No question provided")
            return jsonify({'error': 'No question provided'}), 400

        # Get the PDF text from session
        pdf_text = app.config.get('CURRENT_PDF_TEXT')
        if not pdf_text:
            return jsonify({'error': 'No PDF uploaded'}), 400

        try:
            text_chunks = chunk_text(pdf_text, 1000, 100)
            chunk_embeddings = create_embeddings(openai_client, text_chunks)
            retrieved_chunks = retrieve_relevant_chunks(question, text_chunks, chunk_embeddings, k=5)
            context = "\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])

            with open('prompt.json', 'r') as file:
                prompt = json.load(file)['retrieve_survey_prompt'].format(context=context, question=question)
            response = make_groq_request(groq_client, context, prompt)
            return jsonify({'response': response})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 8000))
    
    print(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=True)
