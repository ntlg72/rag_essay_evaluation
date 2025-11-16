import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from .utils import load_dataset, preprocess_dataset
from .config import Config

def generate_predictions(model_path, test_csv_path, output_path):
    """Genera predicciones para el test set"""
    print("\n" + "="*70)
    print("GENERANDO PREDICCIONES PARA TEST")
    print("="*70)

    # Cargar modelo
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Cargar test data
    test_data = load_dataset(test_csv_path)
    test_dataset = preprocess_dataset(test_data, tokenizer, is_test=True)

    # Inferencia
    training_args = TrainingArguments(
        output_dir="/tmp/predictions",
        per_device_eval_batch_size=16,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(model=model, args=training_args)
    predictions = trainer.predict(test_dataset)

    # Post-procesamiento
    scores = predictions.predictions.squeeze()
    scores_final = np.clip(np.round(scores), 1, 6).astype(int)

    # Guardar
    submission = pd.DataFrame({
        'essay_id': test_data['essay_id'],
        'score': scores_final
    })

    submission.to_csv(output_path, index=False)
    print(f"✅ Predicciones guardadas en: {output_path}")
    print(f"\n{submission.head()}")