from backend.training import setup_model_and_lora, train_model, save_model
from backend.utils import mount_drive, load_dataset, preprocess_dataset
from backend.rag import load_rubrics, create_vector_store
from backend.models import CustomLLM
from transformers import AutoTokenizer
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import Config

def main_training_pipeline():
    print("\n" + "="*70)
    print("PIPELINE DE ENTRENAMIENTO")
    print("="*70)

    if not mount_drive():
        return

    train_data = load_dataset(f"{Config.DRIVE_PATH}/train.csv", limit=17307)
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL)
    model = setup_model_and_lora()
    dataset = preprocess_dataset(train_data, tokenizer)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    trainer = train_model(split["train"], split["test"], model, tokenizer)
    save_model(trainer, tokenizer, model)

    print("\n✅ Pipeline de entrenamiento completado")

def main_evaluation_pipeline():
    print("\n" + "="*70)
    print("PIPELINE DE EVALUACIÓN")
    print("="*70)

    regressor_llm = CustomLLM(Config.MODEL_PATH)
    generative_llm = ChatGoogleGenerativeAI(
        model=Config.GEMINI_MODEL,
        temperature=0,
        google_api_key=Config.GOOGLE_API_KEY
    )

    documents = load_rubrics(Config.DRIVE_PATH)
    vector_store = create_vector_store(documents)

    retriever = None
    if vector_store:
        retriever = vector_store.as_retriever(search_kwargs={"k": Config.TOP_K_RETRIEVAL})

    # Aquí puedes insertar tu nueva lógica de evaluación sin RAGAS
    essay_example = """
    I am a scientist at NASA discussing the "face" on Mars. The face is a landform,
    not created by aliens. There is no evidence of life on Mars. Many landforms on
    Earth look like familiar objects. NASA scientists confirm this is a natural
    geological formation.
    """

    print("\n Evaluando ensayo de ejemplo...")
    # Aquí podrías usar tu propia función de evaluación, por ejemplo:
    # result = evaluate_essay_custom(essay_example, regressor_llm, generative_llm, retriever)

    print("\n✅ Evaluación completada (sin RAGAS)")