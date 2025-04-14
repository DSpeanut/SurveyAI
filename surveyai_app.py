from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import logging
from groq import Groq  
from dotenv import load_dotenv
from llm_agent import make_groq_request
import PyPDF2
from data_chunk import extract_text_from_pdf, chunk_text, create_embeddings, retrieve_relevant_chunks
from openai import OpenAI
from werkzeug.utils import secure_filename
import json
import secrets

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
CORS(app)  # Enable CORS for all routes
app.secret_key = secrets.token_hex(16)  # Required for sessions
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

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
def upload_pdf():
    try:
        logger.debug("Upload route called")
        
        if 'pdf' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['pdf']
        logger.debug(f"Received file: {file.filename}")
        
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
            
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.debug(f"File saved to: {filepath}")
            
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(filepath)
            logger.debug(f"Extracted text length: {len(pdf_text)}")
            
            # Store the text in app config
            app.config['PDF_TEXT'] = pdf_text
            logger.debug("PDF text stored in app config")
            
            return jsonify({'message': 'File uploaded successfully'}), 200
        else:
            logger.error("Invalid file type")
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        logger.debug(f"Received question data: {data}")
        
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No data received'}), 400
            
        question = data.get('question')
        if not question:
            logger.warning("No question provided")
            return jsonify({'error': 'No question provided'}), 400

        # Get the PDF text from app.config
        pdf_text = app.config.get('PDF_TEXT')
        logger.debug(f"Retrieved PDF text from app.config: {bool(pdf_text)}")
        if not pdf_text:
            logger.error("No PDF text found in app.config")
            return jsonify({'error': 'No PDF uploaded or PDF text not available. Please upload a PDF first.'}), 400

        try:
            logger.debug("Starting text chunking")
            text_chunks = chunk_text(pdf_text, 1000, 100)
            logger.debug(f"Created {len(text_chunks)} text chunks")
            
            logger.debug("Creating embeddings")
            chunk_embeddings = create_embeddings(openai_client, text_chunks)
            logger.debug("Embeddings created successfully")
            
            logger.debug("Retrieving relevant chunks")
            retrieved_chunks = retrieve_relevant_chunks(openai_client, question, text_chunks, chunk_embeddings,5)
            logger.debug(f"Retrieved {len(retrieved_chunks)} relevant chunks")
            
            context = "\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
            logger.debug("Context prepared")

            try:
                with open('prompt.json', 'r') as file:
                    prompt = json.load(file)['retrieve_survey_prompt'].format(context=context, question=question)
                logger.debug("Prompt loaded and formatted")
                
                logger.debug("Making Groq request")
                response = make_groq_request(groq_client, context, prompt)
                logger.debug("Groq request successful")
                
                return jsonify({'response': response})
            except json.JSONDecodeError as e:
                logger.error(f"Error reading prompt.json: {str(e)}")
                return jsonify({'error': 'Error reading prompt template'}), 500
            except Exception as e:
                logger.error(f"Error in LLM request: {str(e)}")
                return jsonify({'error': f'Error processing question: {str(e)}'}), 500
                
        except Exception as e:
            logger.error(f"Error in text processing: {str(e)}", exc_info=True)
            return jsonify({'error': f'Error processing text: {str(e)}'}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in ask_question: {str(e)}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 8000))
    
    print(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=True)
