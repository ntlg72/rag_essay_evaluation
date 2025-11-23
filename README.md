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
