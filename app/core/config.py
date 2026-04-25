from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://localhost:5432/content_gen"
    tavily_api_key: str = ""
    tavily_crawl_depth: int = 3
    tavily_max_pages: int = 20

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
