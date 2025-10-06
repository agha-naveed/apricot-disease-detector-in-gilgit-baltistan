from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

class ApricotModel:
    def __init__(self, model_path='model/model.pt'):
        self.model = YOLO(model_path)
        self.classes = [
            'Mite Infestation', 'Bacterial Shot Hole', 'Gummosis',
            'Scabbed Disease', 'Brown Rot', 'Anthracnose',
            'Carposina Sasakii Matsumura Infestation',
            'Aromia Bungii Infestation',
            'Cnidocampa Flavescens Walker Infestation',
            'Hyphantria Cunea Infestation',
            'Chilocorus Rubidus Hope'
        ]

    def predict(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            results = self.model(image)

            if len(results[0].boxes) == 0:
                return {"disease": "No disease detected", "confidence": 0.0}

            box = results[0].boxes[0]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = self.classes[cls_id] if cls_id < len(self.classes) else f"class_{cls_id}"
            return {"disease": label, "confidence": round(conf * 100, 2)}

        except Exception as e:
            return {"error": str(e)}

model = ApricotModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=["GET"])
def detectPage():
    return render_template('main.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    image_bytes = file.read()
    result = model.predict(image_bytes)

    if 'error' in result:
        return jsonify(result), 500

    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
