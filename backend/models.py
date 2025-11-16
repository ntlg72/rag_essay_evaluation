import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import Config

class CustomLLM:
    """Wrapper para modelo regresor DistilBERT"""
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Modelo cargado en {self.device}")

    def predict_score(self, text):
        """Predice score de 1-6"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=Config.MAX_LENGTH
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            score_raw = outputs.logits.squeeze().cpu().item()
            score = int(np.clip(np.round(score_raw), 1, 6))

            return score
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return 3  # Score por defecto