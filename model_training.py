from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.optim import AdamW
from data_preprocessing import ToxicCommentProcessor
import torch
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not os.path.exists('models'):
    os.makedirs('models')

class ToxicCommentClassifier:   
    
    def __init__(self, model_name='distilbert-base-uncased', batch_size=16, learning_rate=2e-5, epochs=3):
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.processor = None
        self.model = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        self.callback = None
        self.progress_callback = None

    def load_data(self, nrows=None):
        self.processor = ToxicCommentProcessor(self.model_name, nrows=nrows)
        data = self.processor.get_data_for_training()
        self.input_ids = data['input_ids']
        self.attention_masks = data['attention_masks']
        self.labels = data['labels']
        self.num_labels = len(data['target_columns'])
        self.max_length = data['max_length']
        self.dataset = TensorDataset(self.input_ids, self.attention_masks, self.labels)
        val_size = int(0.1 * len(self.dataset))
        train_size = len(self.dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size)

    def initialize_model(self):
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        )
        self.model.to(device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(self.train_dataloader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )    
    def train(self):
        for epoch in range(1, self.epochs + 1):
            if self.progress_callback:
                self.progress_callback({"status": "training", "epoch": epoch, "total_epochs": self.epochs, "message": f"Starting epoch {epoch}/{self.epochs}"})
            
            self.model.train()
            total_loss = 0
            batch_count = len(self.train_dataloader)
            
            for i, batch in enumerate(tqdm(self.train_dataloader, desc="Training")):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                self.model.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                # Report progress every 10% of batches or at least once per epoch
                if self.progress_callback and (i % max(int(batch_count/10), 1) == 0):
                    progress = (i / batch_count) * 100
                    self.progress_callback({
                        "status": "training",
                        "epoch": epoch,
                        "total_epochs": self.epochs,
                        "batch": i + 1,
                        "total_batches": batch_count,
                        "progress": progress,
                        "loss": loss.item(),
                        "message": f"Training epoch {epoch}/{self.epochs} - {progress:.1f}% complete"
                    })
            
            avg_train_loss = total_loss / len(self.train_dataloader)
            self.history['train_loss'].append(avg_train_loss)
            
            if self.progress_callback:
                self.progress_callback({
                    "status": "evaluating", 
                    "epoch": epoch, 
                    "total_epochs": self.epochs,
                    "message": f"Evaluating model after epoch {epoch}/{self.epochs}"
                })
            
            val_loss, val_accuracy = self.evaluate()
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            if self.progress_callback:
                self.progress_callback({
                    "status": "epoch_complete", 
                    "epoch": epoch, 
                    "total_epochs": self.epochs,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "message": f"Completed epoch {epoch}/{self.epochs} - validation accuracy: {val_accuracy:.4f}"
                })
            
            if epoch == self.epochs:
                self._save_checkpoint(epoch)
                if self.progress_callback:
                    self.progress_callback({
                        "status": "completed",
                        "message": "Model training completed and saved"
                    })
            
            if self.callback:
                self.callback(epoch, avg_train_loss, val_loss, val_accuracy)

    def evaluate(self):
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                logits = outputs.logits
                preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
        avg_val_loss = val_loss / len(self.val_dataloader)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        self.metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        print(classification_report(all_labels, all_preds, target_names=self.processor.target_columns, zero_division=0))
        return avg_val_loss, accuracy
    
    def _save_checkpoint(self, epoch):
        checkpoint_path = f"models/{self.model_name.replace('/', '_')}_final.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'val_accuracy': self.history['val_accuracy'],
            'metrics': self.metrics if hasattr(self, 'metrics') else None
        }, checkpoint_path)
        print(f"Final model saved to {checkpoint_path}")