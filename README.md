# Tuberculosis Detection Web Application

A Flask-based web application that uses deep learning to detect Tuberculosis from chest X-ray images.

## Features
- Real-time TB detection from chest X-ray images
- User-friendly web interface
- Instant results with visual feedback
- Support for common image formats

## Tech Stack
- Python 3.x
- Flask
- TensorFlow/Keras
- HTML/CSS
- NumPy
- Pillow

## Project Structure
```
AIProject/
├── app.py              # Main Flask application
├── models/
│   └── TBCdetect.h5   # Trained model file
├── static/            # For uploaded images
├── templates/
│   └── index.html    # Web interface template
└── README.md
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/MatthewKainde/TBC-detection-website.git
cd TBC-detection-website
```

2. Set up virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies
```bash
pip install flask tensorflow numpy pillow
```

## Usage
1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to `http://127.0.0.1:5000`
3. Upload a chest X-ray image
4. View the detection results

## Model Information
- Input: Grayscale chest X-ray images (224x224 pixels)
- Output: Binary classification (Normal/Tuberculosis)
- Based on: Deep learning CNN architecture

## License
MIT License

## Author
Matthew Kainde
- GitHub: [@MatthewKainde](https://github.com/MatthewKainde)