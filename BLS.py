import base64
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms as T
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os

# Setup logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class TextRecognitionApp:
    models = ['vitstr']

    def __init__(self):
        self.device = device
        self._model_cache = {}
        self._preprocess = T.Compose([
            T.Resize((32, 128), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        self.preload_models()

    def preload_models(self):
        for model_name in self.models:
            self._get_model(model_name)

    def _get_model(self, name):
        if name not in self._model_cache:
            model = torch.hub.load('baudm/parseq', name, pretrained=True).eval()
            model.to(self.device)
            self._model_cache[name] = model
        return self._model_cache[name]

    @torch.inference_mode()
    def process_image(self, model_name, image_base64):
        try:
            if not image_base64:
                return "", False
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data)).convert('RGB')
            image = self._preprocess(image).unsqueeze(0).to(self.device)
            model = self._get_model(model_name)
            pred = model(image).softmax(-1)
            label, _ = model.tokenizer.decode(pred)
            return label[0], True
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return "", False

recognition_app = TextRecognitionApp()

# NOUVELLE ROUTE POUR BATCH PROCESSING
@app.route('/solve_batch', methods=['POST'])
def solve_batch():
    try:
        data = request.get_json()
        
        if not data or 'images' not in data:
            return jsonify({"error": "Missing 'images' field"}), 400
        
        images = data['images']
        
        if not isinstance(images, list):
            return jsonify({"error": "'images' must be an array"}), 400
        
        results = []
        model_name = 'vitstr'
        
        for idx, image_base64 in enumerate(images):
            recognized_text, success = recognition_app.process_image(model_name, image_base64)
            
            results.append({
                "success": success,
                "text": recognized_text,
                "index": idx
            })
            
            if success:
                print(f"Image {idx + 1}/{len(images)}: {recognized_text}")
        
        return jsonify(results), 200
        
    except Exception as e:
        print(f"Error in solve_batch: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ANCIENNE ROUTE (gardée pour compatibilité)
@app.route('/', methods=['POST', 'GET'])
def handle_request():
    model_name = request.args.get('a', 'vitstr') 
    image_base64 = request.args.get('b', '')
    number_to_compare = request.args.get('n', '')
    recognized_text, valid = recognition_app.process_image(model_name, image_base64) 

    if not valid:
        response = {"status": "error", "message": "Invalid image data"}
    elif recognized_text == number_to_compare:
        response = {"status": "ok", "message": f"Image {number_to_compare} solved"}
    else:
        response = {"status": "not ok", "message": f"Image {number_to_compare} does not match"}

    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)