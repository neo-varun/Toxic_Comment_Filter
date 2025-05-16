import os
import torch
from transformers import DistilBertForSequenceClassification
from data_preprocessing import ToxicCommentProcessor
import copy
import torch.nn.utils.prune as prune

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelOptimizer:

    def __init__(self, model_path='models/distilbert-base-uncased_final.pt', model_name='distilbert-base-uncased'):
        self.model_path = model_path
        self.model_name = model_name
        self.target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.processor = ToxicCommentProcessor(model_name=model_name)
        self.original_model = self._load_original_model()

    def _load_original_model(self):
        model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.target_columns),
            problem_type="multi_label_classification"
        )
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model
    
    def get_model_size(self, model):
        temp_path = 'models/temp_model.pt'
        torch.save(model.state_dict(), temp_path)
        size_bytes = os.path.getsize(temp_path)
        os.remove(temp_path)
        return size_bytes / (1024 * 1024)
    
    def prune_and_quantize(self):
        pruned_model = copy.deepcopy(self.original_model)
        prunable_layers = [(module, 'weight') for name, module in pruned_model.named_modules() if isinstance(module, torch.nn.Linear)]
        prune.global_unstructured(prunable_layers, pruning_method=prune.L1Unstructured, amount=0.3)
        for module, _ in prunable_layers:
            prune.remove(module, 'weight')
        quantized_model = torch.quantization.quantize_dynamic(pruned_model, {torch.nn.Linear}, dtype=torch.qint8)
        return quantized_model
    
    def export_to_onnx(self, model, batch_size=1, sequence_length=128):
        dummy_input = {
            'input_ids': torch.randint(0, self.processor.tokenizer.vocab_size, (batch_size, sequence_length), device=device),
            'attention_mask': torch.ones((batch_size, sequence_length), device=device, dtype=torch.long)
        }
        onnx_path = f'models/{self.model_name.replace("/", "_")}_optimized.onnx'
        try:
            torch.onnx.export(
                model,
                (dummy_input['input_ids'], dummy_input['attention_mask']),
                onnx_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['output'],
                dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}},
                opset_version=13
            )
        except Exception as e:
            print(f"Error exporting to ONNX: {e}")
        return onnx_path
    
    def create_optimized_model(self):
        optimized_model = self.prune_and_quantize()
        optimized_model_path = f'models/{self.model_name.replace("/", "_")}_fully_optimized.pt'
        torch.save({'model_state_dict': optimized_model.state_dict()}, optimized_model_path)
        onnx_path = self.export_to_onnx(optimized_model)
        return optimized_model_path, onnx_path

    def run_all_optimizations(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        original_size = self.get_model_size(self.original_model)
        optimized_model_path, onnx_path = self.create_optimized_model()
        return {
            'original_model_path': self.model_path,
            'optimized_model_path': optimized_model_path,
            'onnx_path': onnx_path,
            'original_size_mb': original_size
        }