# Apricot Disease Detection in Gilgit Baltistan using Deep Learning

**Project by Green Revolution PK**

This project is focused on detecting apricot diseases using state-of-the-art deep learning techniques. It specifically targets apricot orchards in the Gilgit-Baltistan region to help farmers identify diseases early and improve crop yield.

---

## 🧠 Technologies Used

### Backend & Frontend:
- **Flask** – Lightweight web framework for Python
- **HTML** – For the frontend interface

### Machine Learning / Deep Learning:
- **PyTorch** – For model development and training
- **YOLOv8 (You Only Look Once)** – Real-time object detection model for disease detection
- **Matplotlib** – For plotting and visualizing model performance
- **Pillow (PIL)** – Image processing
- **OpenCV-Python** – Computer vision tasks (image handling, preprocessing, etc.)

---

## 🚀 Features

- Upload apricot leaf or fruit images via a simple web interface
- Detect diseases using a trained YOLOv8 deep learning model
- Display detection results with bounding boxes and confidence scores
- Visualize model performance metrics (accuracy, loss, etc.)
- Designed to support local farmers with an accessible tool

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/agha-naveed/apricot-disease-detector-in-gilgit-baltistan
cd apricot-disease-detector-in-gilgit-baltistan

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Usage

```bash
# Run the Flask app
python app.py
```

- Navigate to http://localhost:5000 in your web browser
- Upload an image to detect diseases
