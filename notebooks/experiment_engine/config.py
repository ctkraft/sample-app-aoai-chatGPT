import os
import json

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

    class Config:
        env_file = f"{os.path.dirname(os.path.abspath(__file__))}" + "/.env"
        env_file_encoding = "utf-8"

    def get_version_configs(self):
        with open("version_settings.json") as f:
            version_env = json.load(f)[self.VERSION_ID]["ENV"]
        
        self.AZURE_SEARCH_SERVICE = version_env["AZURE_SEARCH_SERVICE"]
        self.AZURE_SEARCH_INDEX = version_env["AZURE_SEARCH_INDEX"]
        self.AZURE_SEARCH_KEY = version_env["AZURE_SEARCH_KEY"]
        self.AZURE_SEARCH_USE_SEMANTIC_SEARCH = version_env["AZURE_SEARCH_USE_SEMANTIC_SEARCH"]
        self.AZURE_SEARCH_TOP_K = version_env["AZURE_SEARCH_TOP_K"]
        self.AZURE_SEARCH_QUERY_TYPE = version_env["AZURE_SEARCH_QUERY_TYPE"]
        self.AZURE_OPENAI_TEMPERATURE = version_env["AZURE_OPENAI_TEMPERATURE"]
        self.AZURE_OPENAI_TOP_P = version_env["AZURE_OPENAI_TOP_P"]
        self.AZURE_OPENAI_SYSTEM_MESSAGE = version_env["AZURE_OPENAI_SYSTEM_MESSAGE"]
        self.SHOULD_STREAM = True if version_env["AZURE_OPENAI_STREAM"].lower() == "true" else False

settings = Settings()
settings.get_version_configs()