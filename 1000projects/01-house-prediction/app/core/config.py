from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    app_name: str = "House Price Prediction API"
    app_version: str = "1.0.0"
    debug: bool = False

    database_url: str = Field(
        default="sqlite:///./data/app.db",
        description="SQLAlchemy URL (SQLite for local dev, PostgreSQL in production).",
    )

    model_dir: Path = Field(default=ROOT / "models")
    model_filename: str = "house_price_model.joblib"
    metadata_filename: str = "metadata.json"

    cookie_secure: bool = Field(
        default=False,
        description="Set True in production (HTTPS only cookies).",
    )
    cookie_samesite: str = Field(default="lax")

    bootstrap_user_email: str = "demo@example.com"
    bootstrap_user_password: str = "demo-pass-123"
    bootstrap_user_name: str = "Demo User"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / self.metadata_filename


settings = Settings()
