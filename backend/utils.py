import os
import gc
import time
import numpy as np
import pandas as pd
from datasets import Dataset, Value
from transformers import AutoTokenizer
#from google.colab import drive  # Solo en Colab; usa try/except en funciones

from .config import Config

def mount_drive():
    """Monta Google Drive (solo Colab)"""
    try:
        drive.mount('/content/drive', force_remount=True)
        print("✅ Google Drive montado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error montando Drive: {e}")
        return False

def load_dataset(csv_path, limit=None):
    """Carga un dataset CSV con validación"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró: {csv_path}")

    data = pd.read_csv(csv_path)
    if limit:
        data = data.iloc[:limit]

    print(f"✅ Dataset cargado: {len(data)} filas")
    return data

def preprocess_dataset(data, tokenizer, is_test=False):
    """Preprocesa dataset para entrenamiento/inferencia"""
    dataset = Dataset.from_pandas(data)

    if not is_test:
        dataset = dataset.rename_column("score", "labels")

    def tokenize(examples):
        encodings = tokenizer(
            examples["full_text"],
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_LENGTH
        )
        if not is_test:
            encodings["labels"] = [np.float32(x) for x in examples["labels"]]
        return encodings

    dataset = dataset.map(tokenize, batched=True, batch_size=16)

    if not is_test:
        dataset = dataset.cast_column("labels", Value(dtype='float32'))
        columns = ["input_ids", "attention_mask", "labels"]
    else:
        columns = ["input_ids", "attention_mask"]

    dataset.set_format(type="torch", columns=columns, output_all_columns=False)

    return dataset