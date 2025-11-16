import os
import io
import base64
import json
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, redirect, session, jsonify, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_me_in_production'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024
STATS_FILE = 'data/stats.json'

# Gemini API Configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyD_ZChlw6WxeQIgkf__AmSnGE1pJOPY_CY')
GEMINI_API_URL = os.environ.get('GEMINI_API_URL', 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists('data'):
    os.makedirs('data')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load model
model = None
model_layer_name = None

try:
    model = load_model('models/TBCdetect.h5')
    print("✓ Model loaded successfully")
    print(f"Model input shape: {model.input_shape}")
    
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            model_layer_name = layer.name
            print(f"✓ Using layer for Grad-CAM: {model_layer_name}")
            break
    # If no layer name with 'conv' found, try to pick the last layer with 4D output (likely a conv layer)
    if model_layer_name is None:
        for layer in reversed(model.layers):
            try:
                shape = getattr(layer, 'output_shape', None)
                if shape is None:
                    continue
                # output_shape can be (None, H, W, C) or similar
                if isinstance(shape, (list, tuple)) and len(shape) >= 3:
                    # prefer layers with 4 dimensions (batch, H, W, C)
                    if len(shape) == 4:
                        model_layer_name = layer.name
                        print(f"✓ Fallback using layer for Grad-CAM: {model_layer_name}")
                        break
            except Exception:
                continue
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# Chatbot Knowledge Base
CHATBOT_RESPONSES = {
    'en': {
        'hello': 'Hello! I\'m the TB Detection Assistant. How can I help you today?',
        'hi': 'Hi there! Feel free to ask me anything about tuberculosis detection or how to use this application.',
        'what is tb': 'Tuberculosis (TB) is a serious infectious disease caused by the bacteria Mycobacterium tuberculosis. It primarily affects the lungs but can spread to other organs.',
        'how does detection work': 'Our AI model uses a deep learning CNN (Convolutional Neural Network) trained on thousands of chest X-rays. It analyzes the image to detect TB-related abnormalities.',
        'what is grad cam': 'Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that visualizes which parts of the X-ray the AI focuses on to make its prediction. Red areas indicate high importance.',
        'how accurate is this': 'This tool is for educational and screening purposes. Always consult with healthcare professionals for accurate diagnosis. The AI provides preliminary analysis only.',
        'how to upload': 'Click the upload box or drag a chest X-ray image (JPG/PNG format). The image will be processed and analyzed automatically.',
        'what file format': 'We accept JPG and PNG image formats. The image should be at least 50x50 pixels. Maximum file size is 10MB.',
        'symptoms of tb': 'Common TB symptoms include: persistent cough (3+ weeks), chest pain, coughing up blood, fatigue, fever, and night sweats. Seek medical attention if you experience these.',
        'is this medical advice': 'No. This is an educational tool for screening purposes only. Always consult qualified healthcare professionals for medical diagnosis and treatment.',
        'privacy': 'Your uploaded images are processed temporarily for analysis and are not stored permanently. Always review our privacy policy for complete information.',
        'confidence score': 'The confidence score shows how sure the AI is about its prediction. Higher scores (90%+) indicate stronger predictions. Scores below 60% require professional verification.',
        'how to interpret': 'Green badge = Normal (no TB detected). Red badge = TB detected. Check the Grad-CAM heatmap to see which areas influenced the decision.',
        'help': 'I can help you with: TB information, how to use the app, interpreting results, file formats, symptoms, and more. Just ask!',
        'default': 'I\'m not sure about that. You can ask me about: TB detection, how to use this app, interpreting results, symptoms, or general TB information.'
    },
    'id': {
        'hello': 'Halo! Saya adalah Asisten Deteksi TB. Bagaimana saya bisa membantu Anda hari ini?',
        'hi': 'Halo! Silakan tanyakan kepada saya apa saja tentang deteksi tuberkulosis atau cara menggunakan aplikasi ini.',
        'what is tb': 'Tuberkulosis (TB) adalah penyakit menular yang serius yang disebabkan oleh bakteri Mycobacterium tuberculosis. Ini terutama mempengaruhi paru-paru tetapi dapat menyebar ke organ lain.',
        'how does detection work': 'Model AI kami menggunakan CNN (Convolutional Neural Network) pembelajaran mendalam yang dilatih pada ribuan sinar-X dada. Ini menganalisis gambar untuk mendeteksi kelainan yang terkait dengan TB.',
        'what is grad cam': 'Grad-CAM (Gradient-weighted Class Activation Mapping) adalah teknik yang memvisualisasikan bagian mana dari sinar-X yang difokuskan AI untuk membuat prediksinya. Area merah menunjukkan pentingnya tinggi.',
        'how accurate is this': 'Alat ini hanya untuk tujuan pendidikan dan skrining. Selalu konsultasikan dengan profesional kesehatan untuk diagnosis yang akurat. AI hanya memberikan analisis awal.',
        'how to upload': 'Klik kotak unggah atau seret gambar sinar-X dada (format JPG/PNG). Gambar akan diproses dan dianalisis secara otomatis.',
        'what file format': 'Kami menerima format gambar JPG dan PNG. Gambar harus memiliki ukuran setidaknya 50x50 piksel. Ukuran file maksimum adalah 10MB.',
        'symptoms of tb': 'Gejala umum TB meliputi: batuk terus-menerus (3+ minggu), nyeri dada, batuk darah, kelelahan, demam, dan keringat malam. Cari perhatian medis jika Anda mengalami gejala ini.',
        'is this medical advice': 'Tidak. Ini adalah alat pendidikan untuk tujuan skrining saja. Selalu konsultasikan dengan profesional kesehatan yang berkualitas untuk diagnosis dan pengobatan medis.',
        'privacy': 'Gambar yang Anda unggah diproses sementara untuk analisis dan tidak disimpan secara permanen. Selalu tinjau kebijakan privasi kami untuk informasi lengkap.',
        'confidence score': 'Skor kepercayaan menunjukkan seberapa yakin AI terhadap prediksinya. Skor yang lebih tinggi (90%+) menunjukkan prediksi yang lebih kuat. Skor di bawah 60% memerlukan verifikasi profesional.',
        'how to interpret': 'Lencana hijau = Normal (tidak terdeteksi TB). Lencana merah = TB terdeteksi. Periksa peta panas Grad-CAM untuk melihat area mana yang mempengaruhi keputusan.',
        'help': 'Saya dapat membantu Anda dengan: informasi TB, cara menggunakan aplikasi, menginterpretasikan hasil, format file, gejala, dan lainnya. Tinggal tanya saja!',
        'default': 'Saya tidak yakin tentang itu. Anda dapat bertanya kepada saya tentang: deteksi TB, cara menggunakan aplikasi ini, menginterpretasikan hasil, gejala, atau informasi umum tentang TB.'
    }
}

def load_stats():
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading stats: {e}")
    
    return {
        'total_scans': 0,
        'tb_cases': 0,
        'normal_cases': 0,
        'sensitivity': 0,
        'specificity': 0,
        'precision': 0,
        'f1_score': 0
    }

def save_stats(stats):
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving stats: {e}")
        return False

def calculate_accuracy(stats):
    if stats['total_scans'] == 0:
        return 0
    return round((stats['sensitivity'] + stats['specificity']) / 2)

def update_stats(result):
    stats = load_stats()
    stats['total_scans'] += 1
    
    if result == 'Tuberculosis':
        stats['tb_cases'] += 1
    else:
        stats['normal_cases'] += 1
    
    save_stats(stats)
    return stats

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_chest_xray(img_cv):
    """Validate if image is a valid chest X-ray - relaxed validation"""
    try:
        if img_cv is None:
            return False, "Failed to read image"
        
        if len(img_cv.shape) == 3:
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_cv
        
        img_resized = cv2.resize(img_gray, (224, 224))
        
        mean_val = np.mean(img_resized)
        std_dev = np.std(img_resized)
        
        print(f"Image validation - Mean: {mean_val:.2f}, Std: {std_dev:.2f}")
        
        if std_dev < 5:
            return False, "Image lacks contrast"
        
        if mean_val < 10 or mean_val > 245:
            return False, "Image is too dark or too bright"
        
        hist = cv2.calcHist([img_resized], [0], None, [256], [0, 256])
        max_hist = np.max(hist)
        total_pixels = img_resized.size
        
        if max_hist > (total_pixels * 0.6):
            return False, "Image appears to be mostly uniform color"
        
        print("✓ Image passed validation")
        return True, "Valid image"
    except Exception as e:
        print(f"Validation error: {e}")
        return False, f"Validation error: {str(e)}"

def generate_gradcam(model, img_array, layer_name):
    try:
        if layer_name is None:
            return None
            
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.outputs[0]]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            class_channel = predictions[:, class_idx]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

def overlay_gradcam(original_img, heatmap):
    try:
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
        return overlay
    except Exception as e:
        print(f"Overlay error: {e}")
        return original_img

def generate_chart(tb_count, normal_count):
    """Generate a pie chart of predictions"""
    if tb_count == 0 and normal_count == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#641B2E')
    ax.set_facecolor('#641B2E')
    
    colors = ['#BE5B50', '#FBDB93']
    labels = ['TB Detected', 'Normal']
    sizes = [tb_count, normal_count]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, textprops={'color': 'white', 'fontsize': 12})
    
    for autotext in autotexts:
        autotext.set_color('#2b0f10')
        autotext.set_fontweight('bold')
    
    ax.set_title('Prediction Distribution', color='white', fontsize=14, pad=20)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#641B2E', edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{chart_base64}"

@app.route('/set_language/<lang>')
def set_language(lang):
    session['lang'] = lang if lang in ['en', 'id'] else 'en'
    return jsonify({'lang': session['lang']})

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence_pct = None
    image_base64 = None
    gradcam_image = None
    
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No file uploaded', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            if not allowed_file(file.filename):
                flash('Invalid file type. Please upload JPG or PNG images.', 'error')
                return redirect(request.url)
            
            img_data = file.read()
            if not img_data:
                flash('File is empty', 'error')
                return redirect(request.url)
            
            print(f"File received: {file.filename}, Size: {len(img_data)} bytes")
            
            try:
                img_cv = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)
                if img_cv is None:
                    flash('Corrupted or invalid image file.', 'error')
                    return redirect(request.url)
                print(f"Image decoded: shape {img_cv.shape}")
            except Exception as e:
                print(f"Decode error: {e}")
                flash(f'Failed to decode image: {str(e)}', 'error')
                return redirect(request.url)
            
            if img_cv.shape[0] < 50 or img_cv.shape[1] < 50:
                flash('Image is too small. Please upload a larger image (min 50x50px).', 'error')
                return redirect(request.url)
            
            is_valid_xray, validation_msg = validate_chest_xray(img_cv)
            if not is_valid_xray:
                print(f"Validation warning: {validation_msg}")
                flash(f'⚠️ Warning: {validation_msg}', 'warning')
            
            try:
                img = image.load_img(io.BytesIO(img_data), target_size=(224, 224), color_mode='grayscale')
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0
                
                print(f"Image processed: shape {img_array.shape}")
            except Exception as e:
                print(f"Processing error: {e}")
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(request.url)
            
            try:
                prediction = model.predict(img_array, verbose=0)
                print(f"Raw prediction: {prediction}")
                
                if isinstance(prediction, np.ndarray):
                    if prediction.shape[-1] == 2:
                        confidence = prediction[0][1]
                        result = 'Tuberculosis' if confidence > 0.5 else 'Normal'
                        confidence_pct = (confidence * 100) if result == 'Tuberculosis' else ((1 - confidence) * 100)
                    else:
                        confidence = prediction[0][0]
                        result = 'Tuberculosis' if confidence > 0.5 else 'Normal'
                        confidence_pct = (confidence * 100) if result == 'Tuberculosis' else ((1 - confidence) * 100)
                
                print(f"Result: {result} ({confidence_pct:.2f}%)")
                
                if confidence_pct < 60:
                    flash(f'⚠️ Low confidence ({confidence_pct:.1f}%). Results may not be reliable. Consult a medical professional.', 'warning')
                
                update_stats(result)
                
            except Exception as e:
                print(f"Prediction error: {e}")
                flash(f'Error during prediction: {str(e)}', 'error')
                return redirect(request.url)
            
            try:
                original_img = cv2.resize(img_cv, (224, 224))
                
                _, buffer = cv2.imencode('.png', original_img)
                image_base64 = 'data:image/png;base64,' + base64.b64encode(buffer).decode()
                
                heatmap = generate_gradcam(model, img_array, model_layer_name)
                if heatmap is not None:
                    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
                    overlay_img = overlay_gradcam(original_bgr, heatmap)
                    _, buffer = cv2.imencode('.png', overlay_img)
                    gradcam_image = 'data:image/png;base64,' + base64.b64encode(buffer).decode()
            except Exception as e:
                print(f"Visualization error: {e}")
            
        except Exception as e:
            flash(f'Unexpected error: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template(
        'index.html',
        lang=session.get('lang', 'en'),
        result=result,
        confidence_pct=confidence_pct,
        image=image_base64,
        gradcam_image=gradcam_image
    )

@app.route('/learn')
def learn():
    return render_template('learn.html', lang=session.get('lang', 'en'))

@app.route('/dashboard')
def dashboard():
    stats = load_stats()
    total = stats['total_scans']
    stats['accuracy_rate'] = calculate_accuracy(stats) if total > 0 else 0
    stats['total'] = total
    
    # Generate chart for dashboard
    tb_count = stats.get('tb_cases', 0)
    normal_count = stats.get('normal_cases', 0)
    chart_url = generate_chart(tb_count, normal_count)
    stats['chart_url'] = chart_url  # Add this line
    
    return render_template('dashboard.html', lang=session.get('lang', 'en'), stats=stats, chart_url=chart_url)

@app.route('/api/reset-stats', methods=['POST'])
def reset_stats():
    default_stats = {
        'total_scans': 0, 'tb_cases': 0, 'normal_cases': 0,
        'sensitivity': 0, 'specificity': 0, 'precision': 0, 'f1_score': 0
    }
    save_stats(default_stats)
    return jsonify({'message': 'Statistics reset successfully', 'stats': default_stats})

@app.route('/chatbot')
def chatbot():
    result_context = request.args.get('result', '')
    confidence_context = request.args.get('confidence', '')
    return render_template(
        'chatbot.html',
        lang=session.get('lang', 'en'),
        result_context=result_context,
        confidence_context=confidence_context
    )

import re
from unicodedata import normalize

# Intent pattern recognition — lebih fleksibel dari keyword matching
INTENT_PATTERNS = {
    'symptoms': r'\b(gejala|symptom|tanda|sign|ciri|keluhan|sakit|apa saja tanda|apa aja tanda|batuk|demam|nyeri)\b',
    'transmission': r'\b(menular|transmit|penyebaran|spread|penularan|bagaimana cara terinfeksi|caranya tertular|melalui|droplet|kontak|cara tertular)\b',
    'prevention': r'\b(cegah|prevent|vaksin|vaccine|bcg|pencegahan|cara menghindari|hindari|perlindungan|proteksi|bagaimana caranya tidak terinfeksi)\b',
    'treatment': r'\b(obat|treat|pengobatan|penyembuhan|terapi|bagaimana cara menyembuhkan|cara mengobati|obatan|berapa lama|durasi)\b',
    'diagnosis': r'\b(diagnosis|tes|test|xray|x-ray|rontgen|mantoux|igra|darah|lab|pemeriksaan|bagaimana didiagnosis|cara mendeteksi|cek|screening)\b',
    'accuracy': r'\b(akurat|accuracy|confidence|kepercayaan|valid|reliable|seberapa akurat|seakurat apa|bisa dipercaya|valid|tepat)\b',
    'latent_active': r'\b(laten|latent|aktif|active|perbedaan|beda|apa bedanya|bedanya apa|yang aktif|yang laten)\b',
    'what_is_tb': r'\b(apa itu|what is|definisi|definition|pengertian|arti|maksud)\s+(tb|tbc|tuberkulosis|tuberculosis)\b',
    'upload_app': r'\b(upload|unggah|file|gambar|image|bagaimana|how|cara|langkah|step|prosedur|proses)\b',
    'gradcam': r'\b(gradcam|grad-cam|heatmap|visualisasi|visualization|fokus|focus|area mana|bagian mana|highlight)\b',
    'risk': r'\b(risiko|risk|kemungkinan|likelihood|siapa yang berisiko|yang rentan|faktor risiko|peluang)\b',
    'contact': r'\b(kontak|contact|tertular|terinfeksi|exposure|terpapar|aman|safe|seberapa lama|berapa lama|close contact)\b'
}

# Forbidden topics — strict reject
FORBIDDEN_PATTERNS = r'\b(politik|politics|agama|religion|kriminal|crime|korupsi|murder|kill|teroris|terror|sex|seks|porn|judi|gambling|narkoba|drugs)\b'

def normalize_text(text):
    """Normalize text: lowercase, remove extra spaces, remove punctuation"""
    text = text.lower().strip()
    # Remove common Indonesian/English punctuation
    text = re.sub(r'[?!,;:\'"\-()]+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def detect_intent(message):
    """Detect user intent using regex patterns. Return intent name or None."""
    normalized = normalize_text(message)
    
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, normalized, re.IGNORECASE):
            return intent
    
    return None

def is_tbc_related(message):
    """
    Check if message is TB/health related.
    More lenient: accepts valid TB questions, rejects only forbidden topics.
    """
    normalized = normalize_text(message)
    
    # Hard reject: forbidden topics
    if re.search(FORBIDDEN_PATTERNS, normalized, re.IGNORECASE):
        return False
    
    # Accept if intent detected (TB question)
    if detect_intent(message):
        return True
    
    # Accept if contains any TB/health keywords
    tbc_keywords = [
        'tuberculosis', 'tb', 'tbc', 'chest', 'x-ray', 'xray', 'radiograph', 'lung',
        'symptom', 'cough', 'fever', 'health', 'medical', 'disease', 'infection',
        'diagnosis', 'treatment', 'prevention', 'vaccine', 'bcg', 'latent',
        'active', 'respiratory', 'breathing', 'shortness', 'breath', 'pneumonia',
        'tuberkulosis', 'gejala', 'demam', 'batuk', 'kesehatan', 'medis',
        'penyakit', 'infeksi', 'diagnosis', 'pengobatan', 'pencegahan', 'vaksin',
        'paru-paru', 'paru', 'dada', 'sinar-x', 'foto', 'radiograf', 'pernafasan',
        'napas', 'sesak', 'pneumonia', 'obat', 'resisten', 'dokter', 'rumah sakit',
        'klinik', 'pasien', 'kontak', 'paparan', 'penularan', 'menular', 'udara'
    ]
    
    return any(keyword in normalized for keyword in tbc_keywords)


def local_bot_response(message, lang='id'):
    """
    Smart context-aware TB responder using intent detection.
    - Detects user intent from regex patterns
    - Answers the actual question, no repetition
    - Flexible response selection
    - Follows health assistant guidelines
    """
    intent = detect_intent(message)
    responses = CHATBOT_RESPONSES.get(lang, CHATBOT_RESPONSES.get('id'))
    
    # Intent-based routing with context-aware responses
    intent_responses = {
        'symptoms': responses.get('symptoms of tb'),
        'transmission': "TB menular melalui droplet udara saat orang yang terinfeksi batuk atau bersin. Kontak singkat biasanya aman; risiko lebih tinggi pada kontak lama dengan orang yang aktif TB. Penggunaan masker dan ventilasi yang baik membantu mengurangi risiko penularan.",
        'prevention': "Pencegahan TB meliputi: vaksinasi BCG (saat bayi), hindari kontak dekat dengan penderita TB aktif tanpa perlindungan, jaga nutrisi dan kesehatan imun, tidur cukup, dan periksakan diri jika ada gejala mencurigakan.",
        'treatment': "TB aktif disembuhkan dengan kombinasi obat-obatan rutin selama 6 bulan. Kepatuhan minum obat sangat penting; jangan berhenti sendiri untuk menghindari kekebalan obat. Dokter akan menentukan jenis obat berdasarkan tes kepekaan bakteri.",
        'diagnosis': "TB didiagnosis melalui: tes Mantoux (tes kulit), IGRA (tes darah), pemeriksaan sputum di bawah mikroskop, atau Rontgen dada untuk TB aktif. Screening awal dapat menggunakan X-ray dan alat AI seperti aplikasi ini.",
        'accuracy': "Aplikasi ini hanya untuk screening awal; diagnosis pasti harus dilakukan oleh dokter dan radiolog berpengalaman. Hasil AI bukan pengganti konsultasi profesional kesehatan. Selalu konsultasikan dengan fasilitas kesehatan untuk diagnosis yang akurat.",
        'latent_active': "TB laten: bakteri ada tapi tidak aktif—tidak menular, tanpa gejala. TB aktif: bakteri berkembang, menular, dan menimbulkan gejala. Tes darah/kulit deteksi laten; X-ray dan gejala klinis deteksi aktif. Hanya TB aktif yang terlihat di X-ray dada.",
        'what_is_tb': responses.get('what is tb'),
        'upload_app': "Klik area upload atau seret file gambar X-ray dada ke kotak. Format yang diterima: JPG atau PNG (maksimal 10MB). Sistem akan memproses gambar dan menampilkan hasil prediksi dengan visualisasi Grad-CAM.",
        'gradcam': "Grad-CAM menunjukkan area X-ray mana yang paling mempengaruhi prediksi model AI. Warna merah = bobot prediksi tinggi, area yang dianggap model penting. Ini membantu memahami logika keputusan AI, bukan diagnosis definitif.",
        'risk': "Risiko TB tinggi pada: orang dengan kontak lama dengan penderita aktif, pekerja kesehatan, orang dengan imun lemah (HIV/AIDS), penyakit paru kronis, atau malnutrisi. TB dapat menimpa siapa saja, namun deteksi dini dan pengobatan sangat efektif.",
        'contact': "Kontak dengan penderita TB aktif berisiko tertular, terutama pada kontak lama (keluarga, satu rumah). Tetapi risiko dapat diminimalkan dengan: masker, ventilasi baik, isolasi penderita, dan pengobatan tepat waktu."
    }
    
    # Return intent-based response if available
    if intent and intent in intent_responses and intent_responses[intent]:
        return intent_responses[intent]
    
    # Fallback: friendly help prompt
    help_text = responses.get('help') or (
        "Tanya saya tentang: gejala TB, cara penularan, pencegahan, pengobatan, cara pemeriksaan, akurasi aplikasi, atau cara menggunakan alat ini. Apa yang ingin kamu ketahui?"
    )
    return help_text

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        lang = data.get('lang', 'en')
        result_context = data.get('result_context', '')
        confidence_context = data.get('confidence_context', '')
        
        if not user_message:
            return jsonify({'success': False, 'error': 'Message is required'})
        
        # Validate that message is TB-related
        if not is_tbc_related(user_message):
            redirect_msg = (
                "Sorry, I can only answer questions about tuberculosis. Please ask TB-related questions."
                if lang == 'en'
                else "Maaf, saya hanya dapat menjawab pertanyaan seputar Tuberkulosis (TBC). Silakan ajukan pertanyaan yang berkaitan dengan TBC."
            )
            return jsonify({'success': True, 'response': redirect_msg})
        
        # Detect user intent to guide LLM
        intent = detect_intent(user_message)
        intent_hint = f" (User is asking about: {intent})" if intent else ""
        
        # Build context for AI
        context = ""
        if result_context:
            context = f"\n\nKonteks: Pengguna baru saja melakukan pemeriksaan TB dengan hasil: {result_context}"
        if confidence_context:
            context += f" (Kepercayaan model: {confidence_context}%)"
        
        # Improved system prompt: less strict, more focused on answering the question
        system_prompt = (
            "Anda adalah asisten pendidikan kesehatan khusus Tuberkulosis (TB/TBC) yang ramah dan berpengetahuan.\n\n"
            "TUGAS UTAMA:\n"
            "- Jawab pertanyaan pengguna tentang TB secara langsung, jelas, dan akurat.\n"
            "- Jangan mulai dengan definisi TB kecuali diminta; fokus pada apa yang ditanyakan pengguna.\n"
            "- Gunakan Bahasa Indonesia yang formal namun mudah dipahami.\n\n"
            "PANDUAN KONTEN:\n"
            "1. Gejala: jelaskan tanda-tanda TB yang umum dan kapan harus ke dokter.\n"
            "2. Penularan: jelaskan bagaimana TB menyebar dan cara mencegahnya.\n"
            "3. Pencegahan: berikan tips konkret untuk mengurangi risiko terinfeksi.\n"
            "4. Pengobatan: jelaskan bahwa TB bisa disembuhkan dengan obat-obatan rutin 6 bulan.\n"
            "5. Diagnosis/Pemeriksaan: jelaskan metode deteksi (Mantoux, IGRA, X-ray, sputum).\n"
            "6. Aplikasi/Screening: jelaskan fungsi AI hanya untuk screening awal, bukan diagnosis pasti.\n\n"
            "BATASAN & KEAMANAN:\n"
            "- Jangan berikan diagnosis medis pasti. Selalu sarankan konsultasi ke dokter/profesional kesehatan.\n"
            "- Gunakan bahasa hati-hati: 'berdasarkan informasi umum', 'kemungkinan', 'sebaiknya periksa ke dokter'.\n"
            "- Jangan jawab pertanyaan non-medis atau topik sensitif (politik, agama, dll).\n"
            "- Jangan ulang-ulang jawaban yang sama dalam satu percakapan.\n\n"
            "GAYA:\n"
            "- Ramah, suportif, dan tidak menakut-nakuti.\n"
            "- Jawaban ringkas namun informatif (2-4 paragraf max).\n"
            "- Gunakan poin/bullet jika membantu kejelasan.\n"
            "- Hindari jargon medis berlebihan; jelaskan istilah rumit dengan sederhana."
        )
        
        # Prepare request to Gemini API
        headers = {'Content-Type': 'application/json'}
        
        payload = {
            'contents': [
                {
                    'parts': [
                        {'text': system_prompt + context + f"\n\nPertanyaan pengguna: {user_message}{intent_hint}"}
                    ]
                }
            ],
            'generationConfig': {
                'temperature': 0.3,  # Slightly higher for natural variations, but still focused
                'topP': 0.9,
                'topK': 40,
                'maxOutputTokens': 500,
            }
        }
        
        # Call Gemini API
        try:
            response = requests.post(
                f'{GEMINI_API_URL}?key={GEMINI_API_KEY}',
                json=payload,
                headers=headers,
                timeout=20
            )
        except Exception as e:
            print(f"Chat request error: {e}")
            # Try local fallback immediately on network error
            local_resp = local_bot_response(user_message, lang=lang)
            if local_resp:
                return jsonify({'success': True, 'response': local_resp, 'source': 'local_fallback'})
            return jsonify({'success': False, 'error': 'Network error. Please try again.'})

        print(f"Gemini API status: {response.status_code}")
        
        bot_response = None
        if response.status_code == 200:
            try:
                result = response.json()
                # Parse Gemini response
                if isinstance(result, dict) and 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if isinstance(candidate, dict):
                        content = candidate.get('content', {})
                        parts = content.get('parts', []) if isinstance(content, dict) else []
                        if parts and len(parts) > 0:
                            bot_response = parts[0].get('text', '')
                
                if bot_response:
                    print(f"Gemini response (first 200 chars): {bot_response[:200]}")
                    return jsonify({'success': True, 'response': bot_response, 'source': 'gemini'})
            except Exception as e:
                print(f"Error parsing Gemini response: {e}")
        else:
            print(f"Gemini API error status: {response.status_code}, response: {response.text[:500]}")

        # If Gemini failed, try local fallback responder
        try:
            print(f"Falling back to local responder for intent: {intent}")
            local_resp = local_bot_response(user_message, lang=lang)
            if local_resp:
                return jsonify({'success': True, 'response': local_resp, 'source': 'local_fallback'})
        except Exception as e:
            print(f"Local fallback error: {e}")

        # Last resort: return error
        fallback_msg = (
            "I'm unable to process your request right now. Please try again later or ask a different question."
            if lang == 'en'
            else "Saya tidak dapat memproses permintaan Anda saat ini. Silakan coba lagi nanti atau tanyakan pertanyaan lain."
        )
        return jsonify({'success': False, 'error': fallback_msg})
        
    except requests.exceptions.Timeout:
        # Try local fallback on timeout
        data = request.get_json() or {}
        user_message = data.get('message', '')
        lang = data.get('lang', 'en')
        if user_message:
            local_resp = local_bot_response(user_message, lang=lang)
            if local_resp:
                return jsonify({'success': True, 'response': local_resp, 'source': 'local_fallback'})
        error_msg = "Request timeout. Please try again." if lang == 'en' else "Permintaan habis waktu. Silakan coba lagi."
        return jsonify({'success': False, 'error': error_msg})
    except Exception as e:
        print(f"Chat API error: {e}")
        error_msg = f"Error: {str(e)}" if str(e) else "Internal error. Please try again."
        return jsonify({'success': False, 'error': error_msg})

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File is too large. Maximum size is 10MB.', 'error')
    return redirect('/')

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html', lang=session.get('lang', 'en')), 404

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
