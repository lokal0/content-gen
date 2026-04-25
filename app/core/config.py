from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://localhost:5432/content_gen"
    tavily_api_key: str = ""
    tavily_crawl_depth: int = 3
    tavily_max_pages: int = 20
    seo_api_url: str = "http://localhost:3000"
    seo_api_token: str = ""
    api_bearer_token: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}

    @model_validator(mode="after")
    def fix_db_url(self):
        if self.database_url.startswith("postgresql://"):
            self.database_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        self.database_url = self.database_url.replace("sslmode=require", "ssl=require")
        self.database_url = self.database_url.replace("channel_binding=require", "")
        self.database_url = self.database_url.rstrip("&?")
        return self


settings = Settings()
