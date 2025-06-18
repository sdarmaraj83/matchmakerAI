import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from flask import Flask, render_template, request
import spacy
import PyPDF2
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
from transformers import BertTokenizer, BertModel
import torch
from collections import Counter
import pdfplumber
import traceback
import numpy as np

app = Flask(__name__)

# ===== Initialize Global Variables =====
global_results = []
saved_job_description = ""
top_keywords = []
skills_data = {"labels": [], "counts": []}
score_bins = []

# ===== Global Models and Config =====
bert_model = None
bert_tokenizer = None
PROCESSING_TIMEOUT = 300  # seconds
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"matchmaker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(log_file, maxBytes=1024*1024*5, backupCount=5),
            logging.StreamHandler()
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

def load_models():
    global bert_model, bert_tokenizer, nlp
    try:
        logger.info("Loading NLP models...")
        nlp = spacy.load("en_core_web_sm")
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

load_models()

def extract_text_from_pdf(pdf_path):
    text = ""
    methods_used = []
    extraction_attempts = []

    def log_attempt(method, success, error=None):
        extraction_attempts.append({
            'method': method,
            'success': success,
            'error': str(error) if error else None,
            'text_extracted': len(text) if success else 0
        })

    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            if pdf_reader.is_encrypted:
                logger.warning(f"PDF is encrypted: {pdf_path}")
                try:
                    pdf_reader.decrypt('')
                except Exception as e:
                    log_attempt("encryption_check", False, f"Encrypted PDF: {str(e)}")
                    return "", ["encrypted"]
    except Exception as e:
        logger.warning(f"Error checking PDF encryption: {str(e)}")

    try:
        logger.debug(f"Attempting pdfplumber standard extraction on {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text(layout=False) or page.extract_text(layout=True)
                    if page_text:
                        text += page_text + "\n"
                        logger.debug(f"Page {i+1} extracted {len(page_text)} chars")
                    else:
                        logger.debug(f"Page {i+1} returned empty text")
                except Exception as e:
                    logger.debug(f"Page {i+1} extraction failed: {str(e)}")
                    continue

        if text.strip():
            methods_used.append("pdfplumber-standard")
            log_attempt("pdfplumber-standard", True)
            logger.info(f"Success with pdfplumber standard ({len(text)} chars)")
            return text.strip(), methods_used
        else:
            log_attempt("pdfplumber-standard", False, "No text extracted")
            logger.debug("pdfplumber standard extracted no text")
    except Exception as e:
        log_attempt("pdfplumber-standard", False, e)
        logger.error(f"pdfplumber standard failed: {str(e)}")

    text = ""

    try:
        logger.debug(f"Attempting PyPDF2 extraction on {pdf_path}")
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text() or \
                               getattr(page, 'extract_text', lambda: '')() or \
                               (page.get_contents() and str(page.get_contents()))
                    if page_text:
                        text += page_text + "\n"
                        logger.debug(f"Page {i+1} extracted {len(page_text)} chars")
                    else:
                        logger.debug(f"Page {i+1} returned empty text")
                except Exception as e:
                    logger.debug(f"Page {i+1} extraction failed: {str(e)}")
                    continue

        if text.strip():
            methods_used.append("PyPDF2")
            log_attempt("PyPDF2", True)
            logger.info(f"Success with PyPDF2 ({len(text)} chars)")
            return text.strip(), methods_used
        else:
            log_attempt("PyPDF2", False, "No text extracted")
            logger.debug("PyPDF2 extracted no text")
    except Exception as e:
        log_attempt("PyPDF2", False, e)
        logger.error(f"PyPDF2 failed: {str(e)}")

    text = ""

    try:
        logger.debug(f"Attempting OCR fallback for {pdf_path}")
        try:
            import pytesseract
            from pdf2image import convert_from_path
        except ImportError:
            logger.debug("OCR dependencies not available")
            raise Exception("OCR dependencies not installed")

        images = convert_from_path(pdf_path)
        for i, image in enumerate(images):
            try:
                page_text = pytesseract.image_to_string(image)
                if page_text:
                    text += page_text + "\n"
                    logger.debug(f"Page {i+1} OCR extracted {len(page_text)} chars")
                else:
                    logger.debug(f"Page {i+1} OCR returned empty text")
            except Exception as e:
                logger.debug(f"Page {i+1} OCR failed: {str(e)}")
                continue

        if text.strip():
            methods_used.append("OCR")
            log_attempt("OCR", True)
            logger.info(f"Success with OCR ({len(text)} chars)")
            return text.strip(), methods_used
        else:
            log_attempt("OCR", False, "No text extracted")
            logger.debug("OCR extracted no text")
    except Exception as e:
        log_attempt("OCR", False, e)
        logger.debug(f"OCR attempt failed: {str(e)}")

    if not text.strip():
        file_size = os.path.getsize(pdf_path)
        if file_size < 1024:
            logger.error(f"PDF appears too small ({file_size} bytes), might be corrupt")
            return "", ["corrupt"]
        else:
            logger.warning(f"PDF appears to be image-based (no text layers)")
            return "", ["image-based"]

    return text.strip(), methods_used

def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        return " ".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        logger.error(f"DOCX Extraction Error: {e}")
        return ""

def extract_keywords(text, top_n=20):
    try:
        doc = nlp(text.lower())
        keywords = []
        
        # 1. Noun chunks (e.g., "machine learning")
        keywords.extend([chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2])
        
        # 2. Named entities (e.g., "Python")
        keywords.extend([ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 2])
        
        # 3. Simple nouns (added)
        keywords.extend([token.text for token in doc if token.pos_ == "NOUN" and len(token.text) > 2])
        
        # Remove duplicates and count frequency
        freq = Counter(keywords)
        return [kw for kw, _ in freq.most_common(top_n)]
    except Exception as e:
        logger.error(f"Keyword Extraction Error: {e}")
        return []

def get_bert_embeddings(text):
    try:
        inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    except Exception as e:
        logger.error(f"BERT Embedding Error: {e}")
        return None

def convert_to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj
# **********

def calculate_match_score(job_desc, resume_text, threshold=70, keyword_weight=30, exact_weight=15):
    try:
        # Initialize variables to avoid reference errors
        job_keywords = []
        resume_keywords = []
        missing_required = []
        
        # 1. Get BERT embeddings (may fail)
        job_embedding = get_bert_embeddings(job_desc)
        resume_embedding = get_bert_embeddings(resume_text)
        if job_embedding is None or resume_embedding is None:
            logger.error("Embedding calculation failed")
            return None

        # 2. Extract keywords (safe even if embeddings failed)
        job_keywords = extract_keywords(job_desc)
        resume_keywords = extract_keywords(resume_text)
        
        # Debug: Log keywords
        logger.info(f"Job Keywords: {job_keywords}")
        logger.info(f"Resume Keywords: {resume_keywords}")

        # 3. Calculate boosts
        keyword_overlap = len(set(job_keywords) & set(resume_keywords))
        keyword_boost = min(keyword_overlap * (keyword_weight / 10), keyword_weight)
        
        exact_matches = sum(1 for kw in job_keywords if kw.lower() in resume_text.lower())
        exact_boost = min(exact_matches * (exact_weight / 5), exact_weight)
        
        missing_required = [kw for kw in job_keywords if kw.lower() not in resume_text.lower()]

        # 4. Base similarity
        base_similarity = cosine_similarity(job_embedding, resume_embedding)[0][0] * 100

        # 5. Final score (no penalty for missing keywords)
        adjusted_score = max(0, min(100,
            (base_similarity * 0.6) + 
            keyword_boost + 
            exact_boost
        ))

        logger.info(f"Score Breakdown - Base: {base_similarity} | KW: {keyword_boost} | Exact: {exact_boost} | Final: {adjusted_score}")

        return {
            'base_score': base_similarity,
            'keyword_boost': keyword_boost,
            'exact_boost': exact_boost,
            'final_score': adjusted_score,
            'passed': adjusted_score >= threshold,
            'missing_keywords': missing_required,
            'keywords': resume_keywords,
            'detected_domain': None
        }

    except Exception as e:
        logger.error(f"Similarity calculation error: {e}\n{traceback.format_exc()}")
        return None
# *** new
        keyword_overlap = len(set(job_keywords) & set(resume_keywords))
        logger.info(f"Keyword Overlap: {keyword_overlap} terms")  # Debug
        
        keyword_boost = min(keyword_overlap * (keyword_weight / 10), keyword_weight)
        logger.info(f"Keyword Boost: {keyword_boost}")  # Debug
# *** new        
        return convert_to_serializable({
            'base_score': base_similarity,
            'keyword_boost': keyword_boost,
            'exact_boost': exact_boost,
            'final_score': adjusted_score,
            'passed': adjusted_score >= threshold,
            'missing_keywords': missing_required,  # Still returned for UI
            'keywords': resume_keywords,
            'detected_domain': None
        })
    except Exception as e:
        logger.error(f"Similarity calculation error: {e}")
        return None

#*******


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/', methods=['GET', 'POST'])
def index():
    global global_results, saved_job_description, top_keywords, skills_data, score_bins
    if request.method == 'POST':
        try:
            saved_job_description = request.form.get('job_description', '')
            job_title = request.form.get('job_title', '').strip()
            if not saved_job_description.strip():
                raise ValueError("Job description cannot be empty")
            threshold = int(request.form.get('threshold', 70))
            keyword_weight = int(request.form.get('keywordWeight', 30))
            exact_weight = int(request.form.get('exactMatchWeight', 15))
            resume_files = request.files.getlist('resume_files')
            if not resume_files:
                raise ValueError("Please upload at least one resume file")
            upload_dir = "uploads"
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            top_keywords = extract_keywords(saved_job_description)
            processed_resumes = []
            skill_counts = Counter()
            similarity_scores = []
            failed_files = []
            for resume_file in resume_files:
                filename = secure_filename(resume_file.filename)
                if not filename or not allowed_file(filename):
                    failed_files.append(filename)
                    continue
                filepath = os.path.join(upload_dir, filename)
                try:
                    resume_file.save(filepath)
                    resume_text = ""
                    if filename.lower().endswith('.pdf'):
                        resume_text, _ = extract_text_from_pdf(filepath)
                    elif filename.lower().endswith('.docx'):
                        resume_text = extract_text_from_docx(filepath)
                    if not resume_text.strip():
                        failed_files.append(filename)
                        continue
                    match_result = calculate_match_score(
                        saved_job_description,
                        resume_text,
                        threshold,
                        keyword_weight,
                        exact_weight
                    )
                    if match_result:
                        similarity_scores.append(match_result['final_score'])
                        processed_resumes.append({
                            'filename': filename,
                            'similarity': match_result['final_score'],
                            'base_score': match_result['base_score'],
                            'keyword_boost': match_result['keyword_boost'],
                            'exact_boost': match_result['exact_boost'],
                            'missing_keywords': match_result['missing_keywords'],
                            'keywords': match_result['keywords'],  # Include keywords in results
                            'passed': match_result['passed'],
                            'domain': match_result['detected_domain']
                        })
                    text_lower = resume_text.lower()
                    for kw in top_keywords:
                        if kw.lower() in text_lower:
                            skill_counts[kw.lower()] += 1
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    failed_files.append(filename)
                finally:
                    if os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                        except Exception:
                            pass
            top_skills = skill_counts.most_common(10)
            skills_data = {
                "labels": [skill[0] for skill in top_skills],
                "counts": [skill[1] for skill in top_skills]
            }
            score_bins = [0] * 10
            for score in similarity_scores:
                bin_index = min(int(score // 10), 9)
                score_bins[bin_index] += 1

            # Sort processed_resumes by similarity score in descending order
            processed_resumes.sort(key=lambda x: x['similarity'], reverse=True)

            global_results = convert_to_serializable(processed_resumes)
        except Exception as e:
            logger.error(f"Processing Error: {e}")
        return render_template(
            'index.html',
            results=global_results,
            job_description=saved_job_description,
            top_keywords=top_keywords,
            skills_data=convert_to_serializable(skills_data),
            score_bins=convert_to_serializable(score_bins),
            current_threshold=request.form.get('threshold', '70'),
            current_kw_weight=request.form.get('keywordWeight', '30'),
            current_exact_weight=request.form.get('exactMatchWeight', '15'),
            current_job_title=request.form.get('job_title', '')
        )
    else:
        return render_template(
            'index.html',
            results=global_results,
            job_description=saved_job_description,
            top_keywords=top_keywords,
            skills_data=convert_to_serializable(skills_data),
            score_bins=convert_to_serializable(score_bins),
            current_threshold='70',
            current_kw_weight='30',
            current_exact_weight='15',
            current_job_title=''
        )



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
