# Configuración centralizada del proyecto
class Config:
    """Configuración centralizada del proyecto"""
    # Supabase
    SUPABASE_URL = "https://vmiaanocchknegntuftl.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZtaWFhbm9jY2hrbmVnbnR1ZnRsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDExMTAxOSwiZXhwIjoyMDc1Njg3MDE5fQ.q4aKgDZyY8zWQ9RbYVPBaWJS5CUKwopkvq3JFMBiB-Y"
    GOOGLE_API_KEY = "AIzaSyAoa-AeY-wwowVw8ODHc3H0c2pRd7NdB9o"

    # Rutas
    DRIVE_PATH = "./data"
    MODEL_PATH = "./models/fine_tuned_model"

    # Modelo
    BASE_MODEL = "distilbert-base-uncased"
    EMBEDDING_MODEL = "sentence-transformers/distilbert-base-nli-mean-tokens"
    GEMINI_MODEL = "gemini-2.5-flash"

    # Hiperparámetros
    MAX_LENGTH = 128
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 5e-5
    LORA_R = 2
    LORA_ALPHA = 8

    # RAG
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 10