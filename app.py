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
        'f1_score': 0,
        # track confidence aggregation so we can show a meaningful "accuracy"-like metric
        'sum_confidence': 0.0,
        'avg_confidence': 0.0
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
    """
    Calculate a user-visible accuracy-like metric.
    Preference order:
    - If we have an aggregated average confidence from model predictions, use that (0-100)
    - Otherwise fall back to sensitivity/specificity if present
    """
    if stats.get('total_scans', 0) == 0:
        return 0

    avg_conf = stats.get('avg_confidence', 0)
    if avg_conf and avg_conf > 0:
        # already in percent (0-100), round to two decimals
        return round(avg_conf, 2)

    sens = stats.get('sensitivity', 0)
    spec = stats.get('specificity', 0)
    if sens or spec:
        return round((sens + spec) / 2, 2)

    return 0

def update_stats(result, confidence=None):
    """
    Update stored statistics. Optionally pass `confidence` as a percentage (0-100)
    to keep track of average model confidence which we display as an "accuracy"-like metric.
    """
    stats = load_stats()
    stats['total_scans'] = stats.get('total_scans', 0) + 1

    if result == 'Tuberculosis':
        stats['tb_cases'] = stats.get('tb_cases', 0) + 1
    else:
        stats['normal_cases'] = stats.get('normal_cases', 0) + 1

    # Update aggregated confidence
    try:
        if confidence is not None:
            # ensure float and scale sanity
            conf_val = float(confidence)
            # assume incoming confidence is already a percentage (0-100)
            stats['sum_confidence'] = stats.get('sum_confidence', 0.0) + conf_val
            stats['avg_confidence'] = stats['sum_confidence'] / float(stats['total_scans'])
    except Exception as e:
        print(f"Error updating confidence stats: {e}")

    save_stats(stats)
    return stats

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compute_lbp_histogram(img_gray, num_points=8, radius=1, bins=256):
    """
    Compute Local Binary Pattern histogram to capture texture.
    X-rays have characteristic fine texture; natural photos have different texture patterns.
    """
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(img_gray, num_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    return hist / hist.sum()  # Normalize

def compute_frequency_spectrum(img_gray):
    """
    Analyze frequency domain characteristics.
    X-rays have specific frequency patterns (fine details, edges at organ boundaries).
    Natural photos have different frequency signatures (objects, textures).
    """
    fft = np.fft.fft2(img_gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    
    # Compute power in different frequency bands
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    
    # Low frequency (center region)
    low_freq_region = magnitude_spectrum[center_h-30:center_h+30, center_w-30:center_w+30]
    low_freq_power = np.sum(low_freq_region ** 2)
    
    # High frequency (edges)
    high_freq_power = np.sum(magnitude_spectrum ** 2) - low_freq_power
    
    return low_freq_power, high_freq_power

def analyze_intensity_distribution(img_gray):
    """
    Analyze intensity histogram characteristics.
    X-ray images typically have bimodal or specific intensity distributions
    (dark: outside body, mid: air/tissue, bright: bones).
    Natural photos have different distributions.
    """
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    
    # Find peaks and valleys
    # X-rays typically show distinct regions
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks.append((i, hist[i]))
    
    return len(peaks), hist

def detect_color_artifacts(img_color):
    """
    Detect if image contains color information that indicates it's NOT an X-ray.
    X-rays are grayscale (R≈G≈B channels are similar).
    Natural colored photos (people, objects) have distinct R, G, B values.
    
    Returns: (is_grayscale_like: bool, color_variance: float, skin_tone_ratio: float)
    """
    # Resize for faster processing
    img_small = cv2.resize(img_color, (128, 128))
    
    # Split channels (BGR)
    b, g, r = cv2.split(img_small.astype(np.float32))
    
    # Calculate channel differences (X-ray should have minimal differences)
    # For true grayscale images: R ≈ G ≈ B
    rg_diff = np.mean(np.abs(r - g))  # Typically 0-5 for X-rays, 10-30+ for colored photos
    rb_diff = np.mean(np.abs(r - b))
    gb_diff = np.mean(np.abs(g - b))
    
    color_variance = (rg_diff + rb_diff + gb_diff) / 3.0
    
    print(f"Color analysis - RG_diff: {rg_diff:.2f}, RB_diff: {rb_diff:.2f}, GB_diff: {gb_diff:.2f}, Avg: {color_variance:.2f}")
    
    # Detect skin-tone colors in the image
    # Skin tones typically have: R > G > B, and specific ranges
    # Simplified skin detection: R > 95, G > 40, B > 20, AND (R - G) > 15
    skin_pixels = (
        (r > 95) & (g > 40) & (b > 20) & 
        (r - g > 15) & (r - b > 0)
    )
    skin_tone_ratio = np.mean(skin_pixels.astype(float))
    
    print(f"Skin tone detection - Ratio: {skin_tone_ratio:.3f} ({skin_tone_ratio*100:.1f}%)")
    
    # X-rays should have very low color variance and NO skin tones
    # Threshold: color_variance should be < 8 for X-ray, > 12 for colored photos
    is_grayscale_like = color_variance < 10  # More strict: < 10 instead of < 15
    
    return is_grayscale_like, color_variance, skin_tone_ratio

def detect_medical_structure(img_gray):
    """
    Detect anatomical structures typical of chest X-rays:
    - Symmetrical horizontal/vertical structure
    - Presence of rib cage edges
    - Cardiac silhouette region
    """
    h, w = img_gray.shape
    
    # Detect vertical symmetry (chest X-rays are roughly symmetric left-right)
    left_half = img_gray[:, :w//2]
    right_half = np.fliplr(img_gray[:, w//2:])
    
    if right_half.shape == left_half.shape:
        symmetry_score = 1.0 - (np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0)
    else:
        symmetry_score = 0.0
    
    # Detect horizontal structure (mediastinal/cardiac silhouette)
    center_row = img_gray[h//2-30:h//2+30, :]
    horizontal_edges = cv2.Canny(center_row, 30, 100)
    horizontal_edge_density = np.sum(horizontal_edges > 0) / horizontal_edges.size
    
    return symmetry_score, horizontal_edge_density

def is_xray_image(img_color):
    """
    Robust X-ray image validation using multiple sophisticated heuristics.
    
    Approach:
    1. Detect color artifacts (X-rays are grayscale, photos are colored)
    2. Detect skin tone (rejects photos of people)
    3. Convert to grayscale for further analysis
    4. Check intensity distribution (X-rays have distinct ranges)
    5. Analyze texture using LBP (Local Binary Pattern)
    6. Inspect frequency domain (X-rays have specific frequency signatures)
    7. Detect medical structures (symmetry, anatomical markers)
    8. Edge and contrast analysis
    
    This method ALLOWS colored X-rays but REJECTS non-medical images including photos of people.
    
    Returns: (is_valid: bool, message: str, confidence_score: float)
    """
    try:
        if img_color is None:
            return False, "Failed to read image", 0.0
        
        # Ensure color image for processing
        if len(img_color.shape) == 2:
            img_bgr = cv2.cvtColor(img_color, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img_color
        
        # Convert to grayscale for analysis
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (224, 224))
        
        # Initialize scoring
        xray_score = 0.0
        max_score = 0.0
        failure_reasons = []
        
        # ===== FEATURE 0: COLOR ARTIFACT DETECTION (EARLY REJECTION) =====
        # This must pass or image is immediately rejected as non-X-ray
        is_grayscale_like, color_variance, skin_tone_ratio = detect_color_artifacts(img_bgr)
        
        # Reject if image has too much color (likely a colored photo)
        if color_variance > 15:
            print(f"✗ Image rejected: excessive color variance ({color_variance:.2f} > 15)")
            return False, "Image appears to be a colored photo, not a grayscale X-ray", 0.0
        
        # Reject if image has significant skin tone (likely a photo of a person)
        if skin_tone_ratio > 0.05:  # More than 5% skin-tone pixels
            print(f"✗ Image rejected: detected {skin_tone_ratio*100:.1f}% skin tone pixels (likely a photo of a person)")
            return False, "Image appears to be a photo of a person, not a medical X-ray", 0.0
        
        # X-ray should be mostly grayscale
        if color_variance > 10:
            failure_reasons.append("Excessive color variance (not typical X-ray)")
            xray_score += 0  # Don't penalize further if already passed thresholds
        else:
            xray_score += 2.0
            print(f"✓ Color check passed: Low color variance ({color_variance:.2f})")
        max_score += 2.0
        
        # ===== FEATURE 1: Intensity Distribution =====
        mean_val = float(np.mean(img_resized))
        std_dev = float(np.std(img_resized))
        
        # X-rays typically have mean in 50-200 range and moderate std
        if 30 <= mean_val <= 220 and std_dev >= 15:
            xray_score += 1.5
            print(f"✓ Intensity distribution OK: mean={mean_val:.1f}, std={std_dev:.1f}")
        elif std_dev < 15:
            failure_reasons.append("Insufficient contrast (likely uniform image)")
        elif mean_val < 30 or mean_val > 220:
            failure_reasons.append("Unusual brightness levels for X-ray")
        max_score += 1.5
        
        # ===== FEATURE 2: Texture Analysis (LBP) =====
        try:
            from skimage.feature import local_binary_pattern
            lbp_hist = compute_lbp_histogram(img_resized)
            # X-rays have characteristic LBP distributions - relatively uniform with peaks
            # Natural photos have more variance
            lbp_entropy = -np.sum(lbp_hist[lbp_hist > 0] * np.log2(lbp_hist[lbp_hist > 0] + 1e-10))
            # X-rays: entropy ~4-6, Natural photos: entropy ~6-8
            if 3.5 <= lbp_entropy <= 7.5:
                xray_score += 1.5
                print(f"✓ Texture (LBP entropy={lbp_entropy:.2f}) consistent with medical image")
            elif lbp_entropy > 7.5:
                failure_reasons.append("Image has too much textural complexity (likely natural photo)")
            max_score += 1.5
        except ImportError:
            print("⚠ scikit-image not available, skipping LBP analysis")
            max_score += 1.5  # Don't penalize if lib unavailable
        
        # ===== FEATURE 3: Frequency Domain Analysis =====
        try:
            low_freq, high_freq = compute_frequency_spectrum(img_resized)
            if low_freq > 0:
                freq_ratio = high_freq / (low_freq + 1e-10)
                # X-rays: balanced low/high (ratio ~0.5-1.5); Photos: high variation (ratio >2 or <0.3)
                if 0.3 <= freq_ratio <= 2.5:
                    xray_score += 1.5
                    print(f"✓ Frequency distribution OK: ratio={freq_ratio:.2f}")
                elif freq_ratio > 3:
                    failure_reasons.append("Image has too much high-frequency detail (likely natural photo)")
            max_score += 1.5
        except Exception as e:
            print(f"⚠ Frequency analysis error: {e}, skipping")
            max_score += 1.5
        
        # ===== FEATURE 4: Medical Structure Detection =====
        try:
            symmetry, h_edge_density = detect_medical_structure(img_resized)
            # X-rays are symmetric (score >0.55) and have horizontal structure (h_edge >0.02)
            if symmetry > 0.5 and h_edge_density > 0.01:
                xray_score += 1.5
                print(f"✓ Medical structure detected: symmetry={symmetry:.2f}, h_edges={h_edge_density:.3f}")
            elif symmetry < 0.3 and h_edge_density < 0.005:
                failure_reasons.append("No anatomical structure detected (not a medical image)")
            max_score += 1.5
        except Exception as e:
            print(f"⚠ Structure detection error: {e}, skipping")
            max_score += 1.5
        
        # ===== FEATURE 5: Edge and Contrast =====
        edges = cv2.Canny(img_resized, 50, 150)
        edge_pixels = int(np.sum(edges > 0))
        edge_ratio = edge_pixels / float(img_resized.size)
        
        # X-rays: edge ratio 0.008-0.15; Natural detailed photos: >0.20; Blank: <0.002
        if 0.005 <= edge_ratio <= 0.20:
            xray_score += 1.5
            print(f"✓ Edge density OK: {edge_ratio:.4f}")
        elif edge_ratio > 0.25:
            failure_reasons.append("Excessive edge density (likely detailed natural photo)")
        elif edge_ratio < 0.003:
            failure_reasons.append("Insufficient edges (likely blank or very simple image)")
        max_score += 1.5
        
        # ===== FINAL DECISION =====
        confidence = xray_score / max_score if max_score > 0 else 0.0
        
        print(f"\nX-ray validation score: {xray_score:.2f}/{max_score:.2f} (confidence: {confidence:.1%})")
        
        # Threshold: >60% confidence indicates likely X-ray
        if confidence >= 0.60:
            print("✓ Image ACCEPTED as likely chest X-ray")
            return True, "Valid chest X-ray image", confidence
        else:
            reason = " OR ".join(failure_reasons) if failure_reasons else "Failed to meet X-ray criteria"
            print(f"✗ Image REJECTED: {reason}")
            return False, reason, confidence
    
    except Exception as e:
        print(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Validation error: {str(e)}", 0.0

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
                # Decode in color first for comprehensive validation
                img_color = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                if img_color is None:
                    flash('Corrupted or invalid image file.', 'error')
                    return redirect(request.url)

                # Convert to grayscale for model input / visualization
                img_cv = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                print(f"Image decoded: color shape {img_color.shape}, grayscale: {img_cv.shape}")
            except Exception as e:
                print(f"Decode error: {e}")
                flash(f'Failed to decode image: {str(e)}', 'error')
                return redirect(request.url)
            
            if img_cv.shape[0] < 50 or img_cv.shape[1] < 50:
                flash('Image is too small. Please upload a larger image (min 50x50px).', 'error')
                return redirect(request.url)
            
            # Validate using robust X-ray detection
            is_valid_xray, validation_msg, xray_confidence = is_xray_image(img_color)
            if not is_valid_xray:
                print(f"Image rejected: {validation_msg}")
                error_msg = f"❌ This image does not appear to be a chest X-ray. {validation_msg}. Please upload a chest X-ray image (JPG or PNG)."
                flash(error_msg, 'error')
                return redirect(request.url)
            
            print(f"Image accepted as X-ray (confidence: {xray_confidence:.1%})")
            
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
                
                # store stats including model confidence (percentage)
                try:
                    update_stats(result, confidence=confidence_pct if confidence_pct is not None else 0.0)
                except Exception as e:
                    print(f"Error saving stats: {e}")
                
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
    
    # Intent-based routing with Dr. Alex Morgan structured responses
    # Format: Direct Answer → Key Details → Practical Guidance → Disclaimer
    intent_responses = {
        'symptoms': (
            "Gejala TB aktif berkembang secara bertahap selama beberapa minggu hingga bulan.\n\n"
            "**DETAIL KUNCI:**\n"
            "• Batuk yang berlangsung 3+ minggu (bisa dengan dahak/lendir)\n"
            "• Demam, terutama sore/malam hari\n"
            "• Keringat malam (bedanya: membasahi pakaian/sprei)\n"
            "• Nyeri dada, terutama saat bernafas atau batuk\n"
            "• Kelelahan, berat badan menurun, nafsu makan turun\n\n"
            "**PANDUAN PRAKTIS:**\n"
            "Jika mengalami gejala di atas selama 3+ minggu, segera periksakan ke dokter atau klinik terdekat. Diagnosis dini sangat penting untuk hasil pengobatan optimal.\n\n"
            "Konsultasikan dengan profesional kesehatan untuk evaluasi medis yang tepat."
        ),
        'transmission': (
            "TB menular melalui droplet udara (percikan air liur) saat orang terinfeksi batuk, bersin, atau berbicara.\n\n"
            "**DETAIL KUNCI:**\n"
            "• Droplet dapat bertahan hingga 2 meter\n"
            "• Kontak singkat dengan penderita TB aktif relatif aman\n"
            "• Risiko tinggi: kontak lama (keluarga, satu rumah, pekerja kesehatan)\n"
            "• TB TIDAK menular melalui: makanan, air, sentuhan, salaman, berbagi barang\n"
            "• TB laten (tidak aktif) sama sekali tidak menular\n\n"
            "**PANDUAN PRAKTIS:**\n"
            "Jika ada orang dengan TB aktif di sekitar, gunakan masker, pastikan ventilasi baik, dan ikuti nasihat dokter. Sebagian orang yang terpapar tidak semuanya terinfeksi; status imun sangat berpengaruh.\n\n"
            "Konsultasikan dengan profesional kesehatan untuk evaluasi medis yang tepat."
        ),
        'prevention': (
            "Pencegahan TB memerlukan kombinasi vaksinasi, kebersihan, dan perilaku sehat.\n\n"
            "**DETAIL KUNCI:**\n"
            "• Vaksinasi BCG pada bayi (efektif ~80% cegah TB berat)\n"
            "• Hindari kontak dekat dengan penderita TB aktif tanpa perlindungan\n"
            "• Jaga sistem imun: nutrisi baik, tidur cukup, olahraga rutin\n"
            "• Periksakan diri jika ada gejala atau kontak dengan penderita TB\n"
            "• Hindari paparan: tembakau, alkohol berlebihan, polusi udara\n\n"
            "**PANDUAN PRAKTIS:**\n"
            "Gaya hidup sehat dan vaksinasi lengkap adalah fondasi. Jika berkontak dengan penderita, konsultasi dokter untuk LTBI screening (tes darah/kulit) dan tindakan profilaksis jika diperlukan.\n\n"
            "Konsultasikan dengan profesional kesehatan untuk evaluasi medis yang tepat."
        ),
        'treatment': (
            "TB aktif dapat disembuhkan dengan rejimen obat-obatan kombinasi selama 6 bulan.\n\n"
            "**DETAIL KUNCI:**\n"
            "• Pengobatan standar: 4 obat (isoniazid, rifampicin, pyrazinamide, ethambutol) fase intensif 2 bulan\n"
            "• Dilanjutkan 2 obat (isoniazid, rifampicin) fase lanjutan 4 bulan\n"
            "• Kepatuhan minum obat SETIAP HARI sangat penting (jangan terlewat)\n"
            "• Jangan berhenti sendiri sebelum 6 bulan = risiko kambuh dan resistensi obat\n"
            "• Efek samping: mual, gatal, perubahan warna urin (normal pada rifampicin)\n\n"
            "**PANDUAN PRAKTIS:**\n"
            "Ikuti jadwal dokter dengan ketat. Buat pengingat harian untuk minum obat. Laporkan efek samping ke dokter. Tes kepekaan obat (DST) menentukan regimen spesifik jika ada resistensi.\n\n"
            "Konsultasikan dengan profesional kesehatan untuk evaluasi medis yang tepat."
        ),
        'diagnosis': (
            "TB didiagnosis melalui kombinasi pemeriksaan klinis, laboratorium, dan pencitraan.\n\n"
            "**DETAIL KUNCI:**\n"
            "• Tes Mantoux (tes kulit): untuk TB laten\n"
            "• IGRA (tes darah): alternatif tes kulit, lebih akurat untuk laten\n"
            "• Sputum AFB (mikroskop): konfirmasi TB aktif, periksa 3 sampel\n"
            "• X-ray dada: untuk TB aktif paru (tampak infiltrat/cavitas)\n"
            "• TB-LAMP (molecular): deteksi cepat, hasil <2 jam\n"
            "• CT scan: untuk kasus kompleks\n\n"
            "**PANDUAN PRAKTIS:**\n"
            "Screening awal dapat menggunakan X-ray dan AI (seperti aplikasi ini) untuk efisiensi. Namun, diagnosis definitif memerlukan konfirmasi dokter dan laboratorium bersertifikat.\n\n"
            "Konsultasikan dengan profesional kesehatan untuk evaluasi medis yang tepat."
        ),
        'accuracy': (
            "Aplikasi ini dirancang untuk screening awal, bukan diagnosis definitif.\n\n"
            "**DETAIL KUNCI:**\n"
            "• AI model kami dilatih pada ribuan X-ray dada untuk deteksi TB\n"
            "• Akurasi di dataset test: ~92%; namun real-world dapat bervariasi\n"
            "• Faktor yang mempengaruhi: kualitas X-ray, teknik pengambilan, fitur klinis lainnya\n"
            "• Hasil AI bukan pengganti diagnosis dokter/radiolog berpengalaman\n"
            "• Konsultasi profesional SELALU diperlukan untuk keputusan klinis\n\n"
            "**PANDUAN PRAKTIS:**\n"
            "Gunakan hasil aplikasi sebagai alat bantu screening—cepat dan affordable. Jika hasil positif atau mencurigakan, selesaikan diagnosis di fasilitas kesehatan dengan dokter spesialis paru dan radiolog.\n\n"
            "Konsultasikan dengan profesional kesehatan untuk evaluasi medis yang tepat."
        ),
        'latent_active': (
            "TB laten dan TB aktif adalah dua kondisi berbeda yang memerlukan pendekatan berbeda.\n\n"
            "**DETAIL KUNCI:**\n"
            "• **TB Laten (LTBI):** Bakteri ada tapi dormant (tidak aktif), tidak menular, tanpa gejala, hanya deteksi via tes\n"
            "• **TB Aktif:** Bakteri aktif dan berkembang, menular via udara, menimbulkan gejala (batuk, demam, dll)\n"
            "• Tes Mantoux/IGRA deteksi TB laten\n"
            "• X-ray, sputum, gejala klinis deteksi TB aktif\n"
            "• ~5-10% orang dengan TB laten akan progress ke aktif jika tidak diobati\n\n"
            "**PANDUAN PRAKTIS:**\n"
            "TB laten perlu dimonitor dan mungkin butuh pencegahan profilaksis (obat preventif 3-6 bulan) tergantung faktor risiko. TB aktif butuh pengobatan kombinasi full-dose 6 bulan.\n\n"
            "Konsultasikan dengan profesional kesehatan untuk evaluasi medis yang tepat."
        ),
        'what_is_tb': (
            "Tuberkulosis (TB) adalah penyakit infeksi yang disebabkan bakteri Mycobacterium tuberculosis.\n\n"
            "**DETAIL KUNCI:**\n"
            "• Bakteri TB terutama menyerang paru-paru (TB paru), tapi bisa organ lain (TB luar-paru)\n"
            "• Dapat dicegah (vaksinasi BCG), didiagnosis (tes/X-ray), dan disembuhkan (obat 6 bulan)\n"
            "• TB adalah penyakit global: ~10 juta kasus/tahun, 1.5 juta kematian/tahun\n"
            "• Namun risiko kematian RENDAH jika dideteksi dini dan diobati\n"
            "• Stigma TB seharusnya berkurang: TB adalah penyakit yang bisa disembuhkan\n\n"
            "**PANDUAN PRAKTIS:**\n"
            "Jika merasa berisiko TB atau memiliki gejala, periksakan diri. Teknologi AI seperti aplikasi ini membantu deteksi cepat, membuat TB lebih mudah dikelola.\n\n"
            "Konsultasikan dengan profesional kesehatan untuk evaluasi medis yang tepat."
        ),
        'upload_app': (
            "Proses upload X-ray dada ke aplikasi ini sangat sederhana dan user-friendly.\n\n"
            "**LANGKAH PRAKTIS:**\n"
            "1. Buka halaman Deteksi (home page aplikasi)\n"
            "2. Klik area upload atau seret file X-ray ke kotak upload\n"
            "3. Format yang diterima: JPG atau PNG (maksimal 10MB, sebaiknya 1-5MB)\n"
            "4. Sistem akan memproses (~2-5 detik) dan menampilkan hasil prediksi\n"
            "5. Visualisasi Grad-CAM menunjukkan area X-ray yang mempengaruhi prediksi\n\n"
            "**TIPS:**\n"
            "• Pastikan X-ray jelas, tidak blur atau terpotong\n"
            "• Jika ada kesalahan, coba X-ray lain atau periksa ke dokter\n\n"
            "Konsultasikan dengan profesional kesehatan untuk evaluasi medis yang tepat."
        ),
        'gradcam': (
            "Grad-CAM (Gradient-weighted Class Activation Mapping) adalah visualisasi yang menunjukkan area mana dalam X-ray yang paling mempengaruhi keputusan AI.\n\n"
            "**DETAIL KUNCI:**\n"
            "• Warna merah/panas = area dengan kontribusi prediksi tinggi (model fokus di sini)\n"
            "• Warna biru/dingin = area dengan kontribusi rendah\n"
            "• Grad-CAM membantu transparansi AI dan interpretabilitas keputusan\n"
            "• BUKAN diagnosis definitif—hanya menunjukkan logika model\n"
            "• Radiolog profesional punya pengalaman lebih luas untuk interpretasi\n\n"
            "**PANDUAN PRAKTIS:**\n"
            "Gunakan Grad-CAM untuk pemahaman: 'AI fokus di area paru bagian atas karena ada kepadatan tertentu.' Tapi jangan percaya sepenuhnya—konsultasi dokter untuk interpretasi definitif.\n\n"
            "Konsultasikan dengan profesional kesehatan untuk evaluasi medis yang tepat."
        ),
        'risk': (
            "Risiko tertular TB atau perkembangan TB laten menjadi aktif berbeda-beda per individu.\n\n"
            "**KELOMPOK RISIKO TINGGI:**\n"
            "• Kontak lama dengan penderita TB aktif (keluarga, satu rumah)\n"
            "• Pekerja kesehatan tanpa perlindungan\n"
            "• Orang dengan imun lemah (HIV/AIDS, immunosuppressants)\n"
            "• Penyakit paru kronis (PPOK, asma berat, pneumokoniosis)\n"
            "• Malnutrisi, gizi buruk, atau gangguan metabolik\n"
            "• Diabetes, gagal ginjal, kanker\n"
            "• Penyalahgunaan tembakau, alkohol, narkoba\n\n"
            "**PANDUAN PRAKTIS:**\n"
            "TB bisa menimpa siapa saja, tapi faktor risiko di atas memerlukan perhatian lebih. Jika termasuk kelompok risiko, dapatkan screening rutin dan konsultasi dengan dokter untuk strategi pencegahan personal.\n\n"
            "Konsultasikan dengan profesional kesehatan untuk evaluasi medis yang tepat."
        ),
        'contact': (
            "Keamanan kontak dengan penderita TB aktif tergantung durasi, intensitas, dan tindakan pencegahan.\n\n"
            "**DETAIL KUNCI:**\n"
            "• Kontak singkat (<2 jam): risiko relatif rendah\n"
            "• Kontak lama (satu rumah, keluarga, teman dekat >8 jam/hari): risiko signifikan\n"
            "• Penggunaan masker dan ventilasi baik mengurangi risiko drastis\n"
            "• TB aktif paru: sangat menular; TB ekstrapulmonary: tidak menular\n"
            "• Setelah penderita minum obat 2 minggu, tingkat penularan turun drastis\n\n"
            "**PANDUAN PRAKTIS:**\n"
            "Jika kontak dengan penderita TB aktif, segera konsultasi dokter untuk LTBI screening. Jangan panik: TB laten yang terdeteksi dapat dicegah dengan obat profilaksis. Terapkan masker dan ventilasi sambil menunggu hasil tes.\n\n"
            "Konsultasikan dengan profesional kesehatan untuk evaluasi medis yang tepat."
        )
    }
    
    # Return intent-based response if available
    if intent and intent in intent_responses and intent_responses[intent]:
        return intent_responses[intent]
    
    # Fallback: friendly, professional help prompt from Dr. Alex Morgan
    help_text = responses.get('help') or (
        "Sebagai Dr. Alex Morgan, saya di sini untuk menjawab pertanyaan tuberculosis Anda. Silakan tanya tentang:\n"
        "• Gejala TB dan kapan periksakan diri\n"
        "• Cara penularan dan pencegahan TB\n"
        "• Pengobatan dan kepatuhan minum obat\n"
        "• Metode diagnosis dan pemeriksaan\n"
        "• Perbedaan TB laten dan TB aktif\n"
        "• Akurasi dan cara menggunakan aplikasi ini\n"
        "• Faktor risiko TB dan kontak dengan penderita\n\n"
        "Apa yang ingin Anda ketahui tentang tuberculosis?"
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
        
        # ===== DR. ALEX MORGAN - TB SPECIALIST SYSTEM PROMPT =====
        # Advanced, structured prompt with expert TB knowledge and strict domain rules
        system_prompt = (
            "Anda adalah Dr. Alex Morgan, seorang spesialis medis senior dengan pengalaman klinis lebih dari 15 tahun "
            "dalam diagnosis tuberkulosis (TB), pengobatan, pencegahan, dan pendidikan kesehatan masyarakat. "
            "Anda bertugas sebagai asisten ahli untuk chatbot deteksi dan informasi TB.\n\n"
            
            "PERAN UTAMA:\n"
            "Tujuan SATU-SATUNYA Anda adalah menjawab pertanyaan yang HANYA berkaitan dengan Tuberkulosis (TB) atau topik kesehatan yang sangat terkait, seperti:\n"
            "- Gejala, tanda, dan presentasi klinis TB\n"
            "- Metode diagnosa TB (Mantoux, IGRA, X-ray, sputum, TB-LAMP, CT scan paru)\n"
            "- Protokol pengobatan TB, rejimen obat, kepatuhan pengobatan, resistensi obat\n"
            "- Strategi pencegahan TB, vaksinasi BCG, kontrol infeksi, penggunaan APD\n"
            "- Mode penularan TB, epidemiologi, faktor risiko\n"
            "- Infeksi TB laten (LTBI) vs penyakit TB aktif\n"
            "- Kesehatan paru umum yang langsung terkait TB (fungsi paru, gejala pernapasan)\n"
            "- AI/ML dalam deteksi TB dan interpretasi pencitraan medis\n\n"
            
            "ATURAN DOMAIN KETAT:\n"
            "1. TOLAK pertanyaan non-TB dengan pesan SINGKAT dan PROFESIONAL:\n"
            "   'Saya di sini hanya untuk membantu menjawab pertanyaan terkait tuberculosis. Silakan tanyakan sesuatu tentang TB.'\n"
            "2. JANGAN memberikan nasihat tentang penyakit tidak terkait TB (diabetes, kanker, kehamilan, kesehatan mental, dll)\n"
            "3. JANGAN memberikan nasihat hukum, keuangan, politik, atau teknis yang tidak terkait TB\n"
            "4. Jika pertanyaan ambigu, tanyakan UNO pertanyaan klarifikasi tapi tetap dalam domain TB\n\n"
            
            "STRUKTUR RESPONS (gunakan untuk SETIAP jawaban):\n"
            "1. JAWABAN LANGSUNG SINGKAT: 1-2 kalimat menjawab inti pertanyaan\n"
            "2. DETAIL KUNCI: 3-5 poin penting berisi fakta, gejala, risiko, atau prosedur\n"
            "3. PANDUAN PRAKTIS: Saran aksi atau langkah berikutnya\n"
            "4. PENAFIAN SINGKAT: 'Konsultasikan dengan profesional kesehatan untuk evaluasi medis yang tepat.'\n\n"
            
            "NADA & PERILAKU:\n"
            "- Profesional, empatik, berbasis bukti ilmiah\n"
            "- Gunakan bahasa sederhana; jelaskan istilah medis\n"
            "- Hindari menakut-nakuti atau spekulasi\n"
            "- JANGAN berikan diagnosis pribadi (misalnya: 'X-ray Anda menunjukkan pneumonia')\n"
            "- JANGAN kontradiksi pedoman keselamatan medis\n"
            "- Singkat: batasi jawaban menjadi 3-4 paragraf maksimal\n\n"
            
            "KONTEKS TAMBAHAN:\n"
            "- Anda memiliki akses ke model AI deteksi TB yang menganalisis X-ray dada\n"
            "- Peran Anda adalah edukatif, bukan diagnostik\n"
            "- Selalu tekankan pentingnya evaluasi medis profesional\n"
            "- Dukung bahasa Inggris dan Indonesia\n\n"
            
            "CONTOH PENOLAKAN:\n"
            "T: 'Bagaimana cara menyembuhkan sakit kepala?' → 'Saya di sini hanya untuk menjawab pertanyaan terkait TB.'\n"
            "T: 'Jelaskan fisika kuantum.' → 'Saya hanya bisa menjawab pertanyaan tentang tuberkulosis.'\n"
            "T: 'Apakah X-ray saya pneumonia?' → 'Saya tidak bisa mendiagnosis. Konsultasikan dengan profesional kesehatan.'\n\n"
            
            "INSTRUKSI BAHASA:\n"
            "- Respons dalam BAHASA INDONESIA (formal namun mudah dipahami)\n"
            "- Gunakan terminologi medis yang tepat tetapi jelaskan istilah rumit dengan sederhana\n"
            "- Hindari slang medis yang tidak perlu; prioritaskan kejelasan untuk pengguna umum"
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
