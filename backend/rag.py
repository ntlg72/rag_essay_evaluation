import os

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

from .config import Config

def load_rubrics(folder_path):
    """Carga rúbricas desde archivos"""
    documents = []

    for filename in os.listdir(folder_path):
        if filename == "train.csv" or filename.endswith(".csv"):
            continue

        filepath = os.path.join(folder_path, filename)
        if not filepath.endswith(('.txt', '.pdf', '.docx')):
            continue

        try:
            loader = UnstructuredFileLoader(filepath, mode="elements")
            docs = loader.load()

            for i, doc in enumerate(docs):
                doc.metadata["file_name"] = filename
                doc.metadata["chunk_index"] = i
                doc.metadata["source"] = filepath

            documents.extend(docs)
            print(f"✅ {filename}: {len(docs)} fragmentos")
        except Exception as e:
            print(f"❌ Error en {filename}: {e}")

    return documents

def create_vector_store(documents):
    """Crea vector store en Supabase"""
    if not documents:
        print("⚠️ No hay documentos para procesar")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    supabase_client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

    vector_store = SupabaseVectorStore(
        client=supabase_client,
        table_name="documents",
        embedding=embeddings
    )

    vector_store.add_documents(chunks)
    print(f"✅ {len(chunks)} fragmentos insertados en Supabase")

    return vector_store