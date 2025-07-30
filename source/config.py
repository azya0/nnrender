from functools import lru_cache

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    THREADS:            int | None = None
    BASE_DATASET_PATH:  str | None = None

    @field_validator("THREADS")
    def validate_host(cls, value: int | None) -> int:
        if value is None:
            return 12
        
        return value
    
    @field_validator("BASE_DATASET_PATH")
    def validate_port(cls, value: str | None) -> str:
        if value is None:
            return "../dataset_info/base.txt"
        
        return value


@lru_cache
def get_settings() -> Settings:
    load_dotenv("../.env")
    return Settings()
