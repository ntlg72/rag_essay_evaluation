import time
from datasets import Dataset
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import Config
from .models import CustomLLM
from .rag import create_vector_store, load_rubrics

EVAL_PROMPT_TEMPLATE = """
Eres un evaluador experto de ensayos. Usa la rúbrica completa proporcionada para evaluar el ensayo en una escala de 1-6.

RÚBRICA:
{context}

ENSAYO:
{essay}

SCORE DEL MODELO: {score}/6

Proporciona una evaluación estructurada:

1. **Puntuaciones**
   - Puntuación rúbrica: X/6
   - Puntuación modelo: {score}/6

2. **Fortalezas**
   (Cita fragmentos específicos de la rúbrica)

3. **Áreas de mejora**
   (Cita fragmentos específicos de la rúbrica)

4. **Justificación por criterio**
   - Contenido:
   - Organización:
   - Estilo:
   - Mecánica:

5. **Comentarios finales**
   (Resumen y recomendaciones)

Sé objetivo y específico en cada punto.
"""

def create_evaluation_chain(regressor_llm, generative_llm, retriever):
    """Crea cadena híbrida de evaluación"""

    def get_score(inputs):
        essay = inputs["essay"]
        return str(regressor_llm.predict_score(essay))

    def get_context(inputs):
        if retriever is None:
            return "Rúbrica no disponible"

        essay = inputs["essay"]
        docs = retriever.invoke(essay)
        return "\n\n".join([doc.page_content for doc in docs])

    eval_prompt = ChatPromptTemplate.from_template(EVAL_PROMPT_TEMPLATE)

    chain = (
        RunnableParallel({
            "essay": lambda x: x["essay"],
            "score": get_score,
            "context": get_context
        })
        | eval_prompt
        | generative_llm
        | StrOutputParser()
    )

    return chain

def evaluate_essay_with_ragas(essay, qa_chain, retriever, ground_truth=None):
    """Evalúa un ensayo y prepara datos para RAGAS"""

    # Recuperar contextos
    context_texts = []
    if retriever:
        try:
            docs = retriever.invoke(essay)
            context_texts = [doc.page_content for doc in docs]
        except Exception as e:
            print(f"⚠️ Error recuperando contexto: {e}")
            context_texts = ["Context unavailable"]

    # Obtener evaluación
    try:
        result = qa_chain.invoke({"essay": essay})
    except Exception as e:
        print(f"❌ Error en evaluación: {e}")
        result = f"Error: {e}"

    # Ground truth por defecto
    if ground_truth is None:
        ground_truth = "Expected evaluation based on rubric criteria"

    # Formato RAGAS
    ragas_data = {
        "question": [essay],
        "answer": [result],
        "contexts": [context_texts],
        "ground_truth": [ground_truth]
    }

    return ragas_data

def run_ragas_evaluation(ragas_data, generative_llm):
    """Ejecuta evaluación RAGAS con manejo de errores"""
    dataset = Dataset.from_dict(ragas_data)

    metrics = [context_precision, context_recall]
    results = {}

    for metric in metrics:
        print(f"\n📊 Evaluando {metric.name}...")

        for attempt in range(2):
            try:
                scores = evaluate(
                    dataset=dataset,
                    metrics=[metric],
                    llm=generative_llm,
                    embeddings=HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    ),
                    raise_exceptions=False
                )

                df = scores.to_pandas()
                if metric.name in df.columns:
                    results[metric.name] = df[metric.name].iloc[0]
                    print(f"  ✅ {metric.name}: {results[metric.name]:.4f}")
                break

            except Exception as e:
                if attempt == 1:
                    results[metric.name] = "ERROR"
                    print(f"  ❌ Falló después de 2 intentos")
                else:
                    time.sleep(5)

    return results