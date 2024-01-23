import os
import json
from definitions import ROOT_DIR

from pydantic import BaseSettings

class Settings(BaseSettings):
    AZURE_SEARCH_SERVICE: str = ""
    AZURE_SEARCH_INDEX: str = ""
    AZURE_SEARCH_KEY: str = ""
    AZURE_SEARCH_USE_SEMANTIC_SEARCH: str = ""
    AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG: str = ""
    AZURE_SEARCH_INDEX_IS_PRECHUNKED: str = ""
    AZURE_SEARCH_TOP_K: int = 5
    AZURE_SEARCH_ENABLE_IN_DOMAIN: str = ""
    AZURE_SEARCH_ID_COLUMNS: str = ""
    AZURE_SEARCH_CONTENT_COLUMNS: str = ""
    AZURE_SEARCH_FILENAME_COLUMN: str = ""
    AZURE_SEARCH_TITLE_COLUMN: str = ""
    AZURE_SEARCH_URL_COLUMN: str = ""
    AZURE_SEARCH_METADATA_COLUMNS: str = ""
    AZURE_SEARCH_VECTOR_COLUMNS: str = ""
    AZURE_SEARCH_QUERY_TYPE: str = ""
    AZURE_SEARCH_PERMITTED_GROUPS_COLUMN: str = ""
    AZURE_USE_DOC_INTEL: bool = False
    AZURE_DOC_INTEL_KEY: str = ""
    AZURE_DOC_INTEL_ENDPOINT: str = ""
    AZURE_OPENAI_RESOURCE: str = ""
    AZURE_OPENAI_MODEL: str = ""
    AZURE_OPENAI_KEY: str = ""
    AZURE_OPENAI_MODEL_NAME: str = ""
    AZURE_OPENAI_TEMPERATURE: float = 0.8
    AZURE_OPENAI_TOP_P: float = 1.0
    AZURE_OPENAI_MAX_TOKENS: int = 8000
    AZURE_OPENAI_STOP_SEQUENCE: str = ""
    AZURE_OPENAI_SYSTEM_MESSAGE: str = ""
    AZURE_OPENAI_PREVIEW_API_VERSION: str = ""
    AZURE_OPENAI_STREAM: str = ""
    SHOULD_STREAM: bool = True
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_EMBEDDING_ENDPOINT: str = ""
    AZURE_OPENAI_EMBEDDING_KEY: str = ""
    AZURE_COSMOSDB_ACCOUNT: str = ""
    AZURE_COSMOSDB_DATABASE: str = ""
    AZURE_COSMOSDB_CONVERSATIONS_CONTAINER: str = ""
    AZURE_COSMOSDB_ACCOUNT_KEY: str = ""
    VERSION_ID: str = ""
    DATA_PATH: str = ""
    AZURE_LOCATION: str = ""
    AZURE_SUBSCRIPTION_ID: str = ""
    AZURE_RESOURCE_GROUP: str = ""
    AZURE_SEARCH_CHUNK_SIZE: int = 512
    AZURE_SEARCH_TOKEN_OVERLAP: int = 128
    AZURE_SEARCH_VECTOR_CONFIG_NAME: str = "default"
    AZURE_SEARCH_SEMANTIC_CONFIG_NAME: str = "default"
    AZURE_SEARCH_ANALYZER_LANGUAGE: str = "en"
    PREP_CONFIG: dict = {}

    class Config:
        env_file = os.path.join(ROOT_DIR, "vars.env")
        env_file_encoding = "utf-8"

    def get_version_configs(self):
        self.DATA_PATH = os.path.join(ROOT_DIR, "data")
        self.PREP_CONFIG = {
            "data_path": self.DATA_PATH,
            "location": self.AZURE_LOCATION,
            "subscription_id": self.AZURE_SUBSCRIPTION_ID,
            "resource_group": self.AZURE_RESOURCE_GROUP,
            "search_service_name": self.AZURE_SEARCH_SERVICE,
            "index_name": self.AZURE_SEARCH_INDEX,
            "chunk_size": self.AZURE_SEARCH_CHUNK_SIZE,
            "token_overlap": self.AZURE_SEARCH_TOKEN_OVERLAP,
            "vector_config_name": self.AZURE_SEARCH_VECTOR_CONFIG_NAME,
            "semantic_config_name": self.AZURE_SEARCH_SEMANTIC_CONFIG_NAME,
            "language": self.AZURE_SEARCH_ANALYZER_LANGUAGE
        }

settings = Settings()
settings.get_version_configs()