import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import mean_squared_error, cohen_kappa_score

from .config import Config
from .utils import preprocess_dataset

def setup_model_and_lora():
    """Configura modelo base con LoRA"""
    config = AutoConfig.from_pretrained(
        Config.BASE_MODEL,
        num_labels=1,
        problem_type="regression"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.BASE_MODEL,
        config=config
    )

    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, lora_config)
    print("✅ Modelo con LoRA configurado")
    return model

def compute_metrics(eval_pred):
    """Calcula MSE y QWK"""
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    labels = labels.astype(np.float32)

    mse = mean_squared_error(labels, predictions)

    pred_rounded = np.clip(np.round(predictions), 1, 6).astype(int)
    ref_rounded = np.clip(np.round(labels), 1, 6).astype(int)
    qwk = cohen_kappa_score(pred_rounded, ref_rounded, weights="quadratic")

    return {"mse": mse, "qwk": qwk}

def train_model(train_dataset, val_dataset, model, tokenizer):
    """Entrena el modelo"""
    training_args = TrainingArguments(
        output_dir=f"{Config.DRIVE_PATH}/results",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        report_to="none",
        fp16=True,
        gradient_accumulation_steps=2,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("🚀 Iniciando entrenamiento...")
    trainer.train()
    print("✅ Entrenamiento completado")

    return trainer

def save_model(trainer, tokenizer, model):
    """Guarda modelo completo"""
    trainer.save_model(Config.MODEL_PATH)
    tokenizer.save_pretrained(Config.MODEL_PATH)
    model.save_pretrained(Config.MODEL_PATH)
    print(f"✅ Modelo guardado en: {Config.MODEL_PATH}")