from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('models/TBCdetect.h5')  # Changed from 'model' to 'models'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['file']
        img_path = 'static/' + img_file.filename
        img_file.save(img_path)

        # If model expects grayscale images (1 channel)
        img = image.load_img(img_path, target_size=(224, 224), color_mode='grayscale')
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        result = 'Tuberculosis Detected' if pred[0][0] > 0.5 else 'Normal'

        return render_template('index.html', result=result, image=img_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
