# RAG Essay Evaluation
Python
Streamlit
License: MIT
Hugging Face
## Descripción
RAG Essay Evaluation es una aplicación web híbrida para la evaluación automática de ensayos académicos en una escala de 1-6, utilizando técnicas de Retrieval-Augmented Generation (RAG). Combina:

Modelo regresor fine-tuned (DistilBERT con LoRA) para puntuación numérica objetiva.
Vector store RAG (Supabase + embeddings de Sentence Transformers) para recuperar criterios de rúbricas relevantes.
LLM generativo (Gemini API) para feedback cualitativo estructurado (fortalezas, áreas de mejora, justificación por criterio: contenido, organización, estilo, mecánica).

Desarrollado para entornos educativos, resuelve la subjetividad en calificaciones manuales, ofreciendo análisis accionable. Soporta despliegue local (Streamlit) o en cloud (Hugging Face Spaces).

#### Características Principales

Interfaz intuitiva con Streamlit: Ingresa ensayo → Obtén score + feedback.
Parsing dinámico de resultados para listas visuales.
Métricas RAGAS opcionales (context_precision, context_recall).
Modular: Backend separado para ML/RAG, frontend para UI.
Compatible con CPU; GPU opcional para aceleración.
Aquí tienes el README en formato Markdown listo para pegar en GitHub, respetando las convenciones estándar, con títulos, emojis, tablas y bloques de código:

```markdown
# 📚 RAG Essay Evaluation System — Evaluación Automática de Ensayos con RAG + LLMs + Streamlit  

Este proyecto implementa un sistema completo de **evaluación automática de ensayos académicos** utilizando **RAG (Retrieval-Augmented Generation)**, modelos de lenguaje avanzados y una **aplicación web en Streamlit**.  
Incluye módulos para recuperación de información, inferencia, entrenamiento, evaluación, carga de rúbricas y una UI que permite evaluar ensayos en tiempo real desde el navegador.

---

## 🎯 Objetivo del Proyecto

El objetivo principal es construir un pipeline integral capaz de:

- 📥 Cargar y procesar rúbricas académicas (PDF)  
- 🔍 Recuperar información relevante con RAG  
- 🧠 Generar evaluaciones automáticas usando modelos LLM entrenados  
- 📝 Calificar ensayos académicos según rúbricas holísticas  
- 📊 Entrenar y comparar modelos  
- 🌐 Proveer una interfaz web en Streamlit  

Este sistema está diseñado para investigación, desarrollo académico y experimentación en evaluación automática de textos.

---

## 📂 Estructura del Repositorio

```
rag_essay_evaluation-main/
├── app.py               # Aplicación Streamlit (puerto 8501)
├── requirements.txt     # Dependencias del proyecto
├── README.md            # Este archivo
│
├── backend/
│   ├── config.py         # Configuraciones globales
│   ├── evaluation.py     # Métricas de performance
│   ├── inference.py      # Pipeline de inferencia principal
│   ├── models.py         # Carga de modelos y tokenizers
│   ├── pipelines.py      # Pipelines de entrenamiento e inferencia
│   ├── rag.py            # Sistema RAG completo
│   ├── training.py       # Fine-tuning y entrenamiento
│   └── utils.py          # Funciones de utilidad
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── submission.csv
│   ├── sample_submission.csv
│   ├── Rubric_Holistic Essay Scoring.pdf
│   └── ...
│
├── models/
│   ├── tokenizer/        # Tokenizador del modelo
│   └── results/
│       └── checkpoint-XXXX/  # Modelos entrenados
│
└── ...
```

---

## 🧠 Tecnologías Utilizadas

- 🐍 Python 3.10+  
- 🎨 **Streamlit**  
- 🤗 HuggingFace Transformers  
- 🔥 PyTorch  
- 🔍 RAG (FAISS / embeddings)  
- 📘 Pandas, NumPy  
- 🧪 Scikit-learn  
- 📄 Procesamiento de PDF  
- 💾 Checkpoints de modelos LLM  

---

## 📋 Requisitos Previos

Asegúrate de tener instalado:

- Python 3.10+  
- pip  
- Git  
- (Opcional) GPU con CUDA  

---

## 🛠️ Instalación

### 1️⃣ Clonar el repositorio

```
git clone <URL_DE_TU_REPO>
cd rag_essay_evaluation-main
```

### 2️⃣ Crear un entorno virtual

```
python -m venv venv
source venv/bin/activate     # Linux / Mac
venv\Scripts\activate        # Windows
```

### 3️⃣ Instalar dependencias

```
pip install -r requirements.txt
```

---

## 🌐 Ejecutar la Aplicación Streamlit

La aplicación principal se encuentra en:

```
app.py
```

Ejecuta:

```
streamlit run app.py
```

La aplicación estará disponible en:

```
http://localhost:8501
```

La interfaz permite:

- Cargar ensayos  
- Seleccionar o cargar rúbricas  
- Recuperar contexto con RAG  
- Generar puntajes automáticos  
- Mostrar evidencia utilizada por el modelo  

---

## 🚀 Ejecución Avanzada (CLI)

Ejecutar inferencia completa (RAG + modelo):

```
python app.py --essay "Tu ensayo aquí" --rubric data/Rubric_Holistic Essay Scoring.pdf
```

Inferencia simple con LLM:

```
python backend/inference.py --text "Ensayo aquí"
```

Usar el módulo RAG directamente:

```
from backend.rag import RAGPipeline

rag = RAGPipeline()
context = rag.retrieve("Texto del ensayo…")
```

---

## 🎓 Entrenamiento del Modelo

Ejecutar el script:

```
python backend/training.py
```

Este script:

- Tokeniza el dataset  
- Entrena el modelo  
- Genera checkpoints en models/results/  

---

## 📊 Evaluación del Modelo

```
python backend/evaluation.py
```

Métricas generadas:

- RMSE  
- MAE  
- Correlación  
- Comparación predicción vs puntaje real  

---

## 🧩 Explicación de Carpetas

- 🧠 backend/: Core del proyecto: RAG, modelos, entrenamiento, inferencia, evaluación, utilidades.  
- 📁 data/: Datasets, rúbricas, ejemplos de submission.  
- 🤗 models/: Modelos entrenados y tokenizers.  
- 🌐 app.py: Aplicación completa en Streamlit.  

---

## 📌 Notas Importantes

- La app SIEMPRE corre en el puerto 8501 (Streamlit).  
- Los checkpoints deben ir en la carpeta models/results/.  
- Las rúbricas deben estar en data/.  
- Si usas CPU, ajusta parámetros en backend/config.py para evitar desbordes de memoria.  

---
