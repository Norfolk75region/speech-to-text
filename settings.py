"""
Настройки по умолчанию: модель и язык записи.

Лежат в settings.json рядом с программой, чтобы переезжать вместе с папкой.
Пока файла нет, модель подбирается под компьютер, а язык — русский.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"


@dataclass
class Settings:
    model: str | None = None  # None — модель подбирается под компьютер
    language: str = "ru"  # код языка (ru, en...) или auto


def load_settings(valid_models: list[str]) -> Settings:
    """Читает настройки. Испорченный файл или неизвестная модель — не ошибка, берём значения по умолчанию."""
    settings = Settings()
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    if not isinstance(data, dict):
        return settings
    if data.get("model") in valid_models:
        settings.model = data["model"]
    language = data.get("language")
    if isinstance(language, str) and language.strip():
        settings.language = language.strip().lower()
    return settings


def save_settings(settings: Settings) -> bool:
    """Сохраняет настройки. False — записать не получилось (например, папка только для чтения)."""
    try:
        SETTINGS_PATH.write_text(
            json.dumps(asdict(settings), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return False
    return True
