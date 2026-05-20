from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Azure OpenAI
    azure_gpt_api: str
    azure_gpt_endpoint: str

    # External APIs
    cookie_yahoo: str = ""
    tavily_api_key: str = ""

    # Model
    deployment_name: str = "gpt-4o"
    api_version: str = "2024-08-01-preview"
    temperature: float = 0.2

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 2
    chroma_db_path: str = "./chroma_db"

    # Logging — set to DEBUG in development to see tool traces
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
