import logging
import re
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image
from keybert import KeyBERT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize KeyBERT for keyword extraction
kw_model = KeyBERT()

def extract_text(file_path):
    try:
        extension = file_path.rsplit('.', 1)[1].lower()
        text_data = []

        if extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                text_data.append({'type': 'paragraph', 'content': content.strip()})

        elif extension == 'pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text()
                # Basic structure detection
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    # Heuristic: Lines in all caps or ending with a colon are headings
                    if line.isupper() or line.endswith(':'):
                        text_data.append({'type': 'heading', 'content': line})
                    else:
                        text_data.append({'type': 'paragraph', 'content': line})

        elif extension == 'docx':
            doc = Document(file_path)
            current_heading = None
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                # Heuristic: Use paragraph style to detect headings
                if para.style.name.startswith('Heading'):
                    current_heading = text
                    text_data.append({'type': 'heading', 'content': text})
                else:
                    text_data.append({'type': 'paragraph', 'content': text})

        elif extension in ['png', 'jpg', 'jpeg']:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang='eng+hin+spa')
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Basic structure detection for OCR output
                if line.isupper() or len(line.split()) < 5:  # Short lines might be headings
                    text_data.append({'type': 'heading', 'content': line})
                else:
                    text_data.append({'type': 'paragraph', 'content': line})

        return text_data

    except Exception as e:
        logging.error(f"Error in extract_text: {e}")
        raise

def extract_keywords(text, top_n=5):
    try:
        # Clean the text
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Extract keywords using KeyBERT
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),  # Allow unigrams and bigrams
            stop_words='english',
            top_n=top_n,
            use_mmr=True,  # Use Maximal Marginal Relevance for diversity
            diversity=0.7
        )
        # Return only the keywords (not the scores)
        return [keyword for keyword, score in keywords]

    except Exception as e:
        logging.error(f"Error in extract_keywords: {e}")
        raise