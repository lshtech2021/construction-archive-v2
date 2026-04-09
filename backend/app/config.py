from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str = ""

    # Azure Document Intelligence
    azure_document_intelligence_endpoint: str = ""
    azure_document_intelligence_key: str = ""

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/construction_archive"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "construction_blueprints"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # ColPali
    colpali_model_name: str = "vidore/colpali-v1.2"
    colpali_device: str = "cpu"

    # Storage
    storage_backend: str = "local"
    local_storage_path: str = "./archive_output"

    # FastAPI
    secret_key: str = "change-me-in-production"
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
