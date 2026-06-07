"""애플리케이션 설정 — pydantic-settings으로 .env에서 로드."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # AWS
    aws_region: str = "ap-northeast-2"
    s3_bucket_raw: str | None = None
    s3_bucket_artifacts: str | None = None
    lancedb_uri: str | None = None

    # LLM / 생성 (Amazon Bedrock)
    bedrock_model_id: str | None = None
    bedrock_rerank_model: str | None = None

    # 임베딩 (Jina-v3)
    jina_api_key: str | None = None
    embed_model: str = "jinaai/jina-embeddings-v3"
    embed_dim: int = 1024

    # PDF 파싱 폴백 (VLM)
    gemini_api_key: str | None = None

    # 메타 보강
    unpaywall_email: str = "cte5301@gmail.com"
    crossref_mailto: str = "cte5301@gmail.com"

    # 서빙
    api_port: int = 8110
    cognito_user_pool_id: str | None = None
    cognito_client_id: str | None = None

    # 평가
    ragas_llm_model: str | None = None

    # 로깅
    log_level: str = "INFO"
