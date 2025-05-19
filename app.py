from flask import Flask, render_template, jsonify, request  # add request
import threading
import os
import json
from model_training import ToxicCommentClassifier
from model_optimization import ModelOptimizer
from transformers import DistilBertForSequenceClassification
import torch
import numpy as np
import onnxruntime as ort
from data_preprocessing import ToxicCommentProcessor  # add this import

app = Flask(__name__)

# Training status and optimization status
training_status = {'running': False, 'message': ''}
optimization_status = {'running': False, 'message': '', 'result': None}

def train_model_thread():
    global training_status
    training_status['running'] = True
    training_status['message'] = 'Training in progress...'
    try:
        classifier = ToxicCommentClassifier()
        classifier.load_data(nrows=1000)
        classifier.initialize_model()
        classifier.train()
        # Save metrics to metrics/metrics.json
        if not os.path.exists('metrics'):
            os.makedirs('metrics')
        with open('metrics/metrics.json', 'w') as f:
            json.dump(classifier.metrics, f)
        training_status['message'] = 'Training completed.'
    except Exception as e:
        training_status['message'] = f'Error: {str(e)}'
    finally:
        training_status['running'] = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    if training_status['running']:
        return jsonify({'status': 'error', 'message': 'Training already in progress.'}), 409
    thread = threading.Thread(target=train_model_thread)
    thread.start()
    return jsonify({'status': 'started', 'message': 'Training started.'})

@app.route('/metrics', methods=['GET'])
def metrics():
    metrics_path = 'metrics/metrics.json'
    if not os.path.exists(metrics_path):
        return jsonify({'status': 'error', 'message': 'No metrics available. Train the model first.'}), 404
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    return jsonify({'status': 'ok', 'metrics': metrics_data})

@app.route('/status', methods=['GET'])
def status():
    return jsonify(training_status)

def optimize_model_thread():
    global optimization_status
    optimization_status['running'] = True
    optimization_status['message'] = 'Optimizing model...'
    try:
        optimizer = ModelOptimizer()
        result = optimizer.run_all_optimizations()
        optimization_status['result'] = result
        optimization_status['message'] = 'Model optimization completed.'
    except Exception as e:
        optimization_status['message'] = f'Error: {str(e)}'
    finally:
        optimization_status['running'] = False

@app.route('/optimize', methods=['POST'])
def optimize():
    if optimization_status['running']:
        return jsonify({'status': 'error', 'message': 'Optimization already in progress.'}), 409
    thread = threading.Thread(target=optimize_model_thread)
    thread.start()
    return jsonify({'status': 'started', 'message': 'Model optimization started.'})

@app.route('/optimization_status', methods=['GET'])
def get_optimization_status():
    return jsonify(optimization_status)

# Add a helper to load ONNX model and run prediction
onnx_session = None
onnx_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def get_onnx_session():
    global onnx_session
    if onnx_session is None:
        onnx_path = 'models/distilbert-base-uncased_optimized.onnx'
        if not os.path.exists(onnx_path):
            raise FileNotFoundError('ONNX model not found. Please optimize the model first.')
        onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    return onnx_session

def predict_pytorch(text):
    model_path = 'models/distilbert-base-uncased_final.pt'
    if not os.path.exists(model_path):
        return None, 'PyTorch model not found. Please train the model first.'
    processor = ToxicCommentClassifier().processor or ToxicCommentProcessor()
    tokenizer = processor.tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=6,
        problem_type="multi_label_classification"
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=processor.max_length)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    return dict(zip(processor.target_columns, probs.round(3).tolist())), None

def predict_onnx(text):
    processor = ToxicCommentProcessor()
    tokenizer = processor.tokenizer
    session = get_onnx_session()
    inputs = tokenizer(text, return_tensors='np', padding='max_length', truncation=True, max_length=processor.max_length)
    ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
    logits = session.run(None, ort_inputs)[0]
    probs = 1 / (1 + np.exp(-logits))[0]
    return dict(zip(onnx_labels, np.round(probs, 3).tolist())), None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    model_type = data.get('model', 'pytorch')
    if not text:
        return jsonify({'status': 'error', 'message': 'No input text provided.'}), 400
    try:
        if model_type == 'onnx':
            prediction, error = predict_onnx(text)
        else:
            prediction, error = predict_pytorch(text)
        if error:
            return jsonify({'status': 'error', 'message': error}), 400
        return jsonify({'status': 'ok', 'prediction': prediction, 'message': 'Prediction complete.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/compare_models', methods=['GET'])
def compare_models():
    import time
    results = []
    # PyTorch model
    pt_path = 'models/distilbert-base-uncased_final.pt'
    pt_size = os.path.getsize(pt_path) / (1024 * 1024) if os.path.exists(pt_path) else None
    pt_time = None
    if pt_size is not None:
        try:
            processor = ToxicCommentProcessor()
            tokenizer = processor.tokenizer
            model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased', num_labels=6, problem_type="multi_label_classification"
            )
            checkpoint = torch.load(pt_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            dummy = tokenizer("This is a test comment.", return_tensors='pt', padding='max_length', truncation=True, max_length=128)
            with torch.no_grad():
                # Warmup
                for _ in range(2):
                    _ = model(**dummy)
                start = time.time()
                for _ in range(20):
                    _ = model(**dummy)
                end = time.time()
                pt_time = ((end - start) / 20) * 1000  # ms
        except Exception as e:
            pt_time = None
    results.append({
        'model': 'PyTorch',
        'size_mb': f"{pt_size:.2f}" if pt_size is not None else 'N/A',
        'inference_ms': f"{pt_time:.2f}" if pt_time is not None else 'N/A'
    })
    # ONNX model
    onnx_path = 'models/distilbert-base-uncased_optimized.onnx'
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024) if os.path.exists(onnx_path) else None
    onnx_time = None
    if onnx_size is not None:
        try:
            import onnxruntime as ort
            processor = ToxicCommentProcessor()
            tokenizer = processor.tokenizer
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            dummy = tokenizer("This is a test comment.", return_tensors='np', padding='max_length', truncation=True, max_length=128)
            ort_inputs = {k: v.astype(np.int64) for k, v in dummy.items() if k in ["input_ids", "attention_mask"]}
            # Warmup
            for _ in range(2):
                _ = session.run(None, ort_inputs)
            import time
            start = time.time()
            for _ in range(20):
                _ = session.run(None, ort_inputs)
            end = time.time()
            onnx_time = ((end - start) / 20) * 1000  # ms
        except Exception as e:
            onnx_time = str(e)
    results.append({
        'model': 'ONNX',
        'size_mb': f"{onnx_size:.2f}" if onnx_size is not None else 'N/A',
        'inference_ms': f"{onnx_time:.2f}" if isinstance(onnx_time, float) else (onnx_time or 'N/A')
    })
    if all(r['size_mb'] == 'N/A' for r in results):
        return jsonify({'status': 'error', 'message': 'No models found to compare.'}), 404
    return jsonify({'status': 'ok', 'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
