import os
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from utils import extract_text, extract_keywords
from summarizers import generate_structured_summary

# Set seed for langdetect to ensure consistent results
DetectorFactory.seed = 0

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()  # Log to console as well
    ]
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        # Get form data
        text_input = request.form.get('text_input', '').strip()
        language = request.form.get('language', 'auto')
        summary_percentage = float(request.form.get('summary_length', 30)) / 100

        # Validate inputs
        if not text_input and 'file_upload' not in request.files:
            return jsonify({'error': 'No file uploaded or text input provided.'}), 400

        # Handle file uploads
        combined_text_data = []
        if 'file_upload' in request.files:
            files = request.files.getlist('file_upload')
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    try:
                        text_data = extract_text(file_path)
                        combined_text_data.extend(text_data)
                    except Exception as e:
                        logging.error(f"Error extracting text from {filename}: {e}")
                        return jsonify({'error': f"Error extracting text from {filename}: {str(e)}"}), 500
                    finally:
                        if os.path.exists(file_path):
                            os.remove(file_path)

        # Add text input to combined data if provided
        if text_input:
            combined_text_data.append({'type': 'paragraph', 'content': text_input})

        if not combined_text_data:
            return jsonify({'error': 'No valid text extracted from inputs.'}), 400

        # Combine all text for processing
        full_text = " ".join(item['content'] for item in combined_text_data)

        # Detect language if set to auto
        if language == 'auto':
            try:
                language = detect(full_text)
                if language not in ['en', 'hi', 'es']:
                    language = 'en'  # Default to English if language not supported
            except Exception as e:
                logging.warning(f"Language detection failed: {e}. Defaulting to English.")
                language = 'en'

        # Translate to English if needed
        if language != 'en':
            try:
                translator = GoogleTranslator(source=language, target='en')
                full_text = translator.translate(full_text)
                logging.info(f"Translated text to English: {full_text[:100]}...")
                # Update combined_text_data with translated text
                for item in combined_text_data:
                    item['content'] = translator.translate(item['content'])
            except Exception as e:
                logging.error(f"Translation error: {e}")
                return jsonify({'error': f"Translation error: {str(e)}"}), 500

        # Generate summary
        try:
            summary = generate_structured_summary(full_text, summary_percentage, combined_text_data)
            if not summary:
                return jsonify({'error': 'Failed to generate summary.'}), 500
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return jsonify({'error': f"Error generating summary: {str(e)}"}), 500

        # Extract keywords
        try:
            keywords = extract_keywords(full_text)
            if not keywords:
                logging.warning("No keywords extracted.")
        except Exception as e:
            logging.error(f"Error extracting keywords: {e}")
            keywords = []

        # Translate summary and keywords back to the original language if needed
        if language != 'en':
            try:
                translator = GoogleTranslator(source='en', target=language)
                summary = translator.translate(summary)
                keywords = [translator.translate(keyword) for keyword in keywords]
                logging.info(f"Translated output back to {language}: Summary - {summary[:100]}...")
            except Exception as e:
                logging.error(f"Error translating back to {language}: {e}")
                return jsonify({'error': f"Error translating output: {str(e)}"}), 500

        return jsonify({'summary': summary, 'keywords': keywords})

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=False)