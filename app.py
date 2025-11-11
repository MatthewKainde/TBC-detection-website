from flask import Flask, request, render_template, jsonify, session
from flask_babel import Babel, gettext as _
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Babel configuration
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_SUPPORTED_LOCALES'] = ['en', 'id']
babel = Babel(app)

# Load model
model = load_model('models/TBCdetect.h5')

# Store predictions in memory (use database in production)
predictions_log = []

def get_locale():
    return session.get('language', 'en')

babel.init_app(app, locale_selector=get_locale)

@app.route('/set_language/<lang>')
def set_language(lang):
    if lang in ['en', 'id']:
        session['language'] = lang
    return jsonify({'status': 'success', 'language': lang})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['file']
        img_path = 'static/uploads/' + img_file.filename
        
        # Create uploads directory if it doesn't exist
        os.makedirs('static/uploads', exist_ok=True)
        img_file.save(img_path)
        
        # Process image
        img = image.load_img(img_path, target_size=(224, 224), color_mode='grayscale')
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        pred = model.predict(img_array)
        confidence = float(pred[0][0])
        confidence_pct = confidence * 100.0 if confidence <= 1.0 else confidence
        result = 'Tuberculosis Detected' if confidence > 0.5 else 'Normal'
        
        # Log prediction
        predictions_log.append({
            'timestamp': datetime.now().isoformat(),
            'result': result,
            'confidence': confidence_pct,
            'filename': img_file.filename
        })
        
        return render_template('index.html',
                               result=result,
                               image=img_path,
                               confidence=confidence,
                               confidence_pct=confidence_pct,
                               lang=get_locale())
    
    return render_template('index.html', lang=get_locale())

@app.route('/learn')
def learn():
    return render_template('learn.html', lang=get_locale())

@app.route('/dashboard')
def dashboard():
    # Calculate statistics
    total_predictions = len(predictions_log)
    tb_count = sum(1 for p in predictions_log if p['result'] == 'Tuberculosis Detected')
    normal_count = total_predictions - tb_count
    avg_confidence = sum(p['confidence'] for p in predictions_log) / total_predictions if total_predictions > 0 else 0
    
    # Generate chart
    chart_url = generate_chart(tb_count, normal_count)
    
    stats = {
        'total': total_predictions,
        'tb_count': tb_count,
        'normal_count': normal_count,
        'tb_percentage': (tb_count / total_predictions * 100) if total_predictions > 0 else 0,
        'avg_confidence': avg_confidence,
        'chart_url': chart_url
    }
    
    return render_template('dashboard.html', stats=stats, lang=get_locale())

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
    
    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#641B2E', edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{chart_base64}"

if __name__ == '__main__':
    app.run(debug=True)