"""
Распознавание речи из аудио- и видеофайлов в текст (faster-whisper).

Без аргументов скрипт работает в пошаговом режиме: спрашивает файл, модель
и сколько человек говорит. С файлами в аргументах сразу распознаёт их:

    python main.py запись.m4a
    python main.py интервью.mp3 --speakers 2 --words
    python main.py лекция.mp4 --model small --device cpu --speakers 1
    python main.py --list-models

Рядом с записью появляются <имя>_transcript.txt (текст с временем и говорящими),
<имя>.srt (субтитры) и с --words <имя>_words.json (время каждого слова).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Прячем служебные предупреждения Hugging Face (например, про HF_TOKEN),
# чтобы они не пугали при скачивании модели. Задаётся до импорта библиотек.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import ctranslate2
from faster_whisper import WhisperModel, decode_audio

from diarization import SAMPLE_RATE, Diarizer, SpeakerLookup
from formats import TextWriter, build_phrases, speaker_label, write_json, write_srt

# Модели скачиваются в папку рядом со скриптом: так её можно целиком
# скопировать на компьютер без интернета.
MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size: str
    description: str


MODELS = [
    ModelInfo("tiny", "75 МБ", "очень быстро, но много ошибок"),
    ModelInfo("base", "145 МБ", "быстро, качество посредственное"),
    ModelInfo("small", "490 МБ", "хороший выбор для компьютера без видеокарты"),
    ModelInfo("medium", "1,5 ГБ", "точнее, но без видеокарты медленно"),
    ModelInfo("large-v3-turbo", "1,6 ГБ", "отлично и быстро на видеокарте"),
    ModelInfo("large-v3", "3 ГБ", "максимальная точность, самая медленная"),
]
MODEL_NAMES = [model.name for model in MODELS]


@dataclass
class Hardware:
    """Что удалось узнать о компьютере."""

    has_gpu: bool
    gpu_name: str | None = None
    gpu_memory_mb: int | None = None

    @property
    def device(self) -> str:
        return "cuda" if self.has_gpu else "cpu"

    def describe(self) -> str:
        if not self.has_gpu:
            return "видеокарта NVIDIA не найдена, работаю на процессоре"
        name = self.gpu_name or "видеокарта NVIDIA"
        if self.gpu_memory_mb:
            return f"{name}, видеопамять {self.gpu_memory_mb / 1024:.0f} ГБ"
        return name

    def recommended_model(self) -> str:
        if not self.has_gpu:
            return "small"
        # Если объём видеопамяти неизвестен, считаем, что её хватает.
        if self.gpu_memory_mb is not None and self.gpu_memory_mb < 3000:
            return "small"
        return "large-v3-turbo"


def detect_hardware() -> Hardware:
    """Проверяет, есть ли видеокарта NVIDIA, которую видит ctranslate2."""
    try:
        has_gpu = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        has_gpu = False
    if not has_gpu:
        return Hardware(has_gpu=False)

    hardware = Hardware(has_gpu=True)
    # Название и объём памяти — только для подсказки, без них тоже работаем.
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            output = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout
            name, memory = output.splitlines()[0].rsplit(",", 1)
            hardware.gpu_name = name.strip()
            hardware.gpu_memory_mb = int(memory.strip())
        except Exception:
            pass
    return hardware


def pick_compute_type(device: str) -> str:
    """Выбирает самый экономный формат вычислений, который поддерживает устройство."""
    supported = ctranslate2.get_supported_compute_types(device)
    preferred = (
        ("int8_float16", "float16", "int8", "float32")
        if device == "cuda"
        else ("int8", "float32")
    )
    for compute_type in preferred:
        if compute_type in supported:
            return compute_type
    return "default"


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class Transcriber:
    """Держит загруженную модель, чтобы не грузить её заново для каждого файла."""

    def __init__(self, model_name: str, device: str, language: str | None):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.model: WhisperModel | None = None
        self.diarizer: Diarizer | None = None

    def _load(self) -> None:
        compute_type = pick_compute_type(self.device)
        where = "видеокарте" if self.device == "cuda" else "процессоре"
        print(f"\nЗагружаю модель {self.model_name} на {where}...")
        if not any(MODELS_DIR.glob(f"models--*--faster-whisper-{self.model_name}")):
            print("Модель скачивается из интернета в первый раз, это может занять несколько минут.")
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=compute_type,
            download_root=str(MODELS_DIR),
        )

    def _switch_to_cpu(self, error: Exception) -> None:
        print(f"\nНа видеокарте не получилось: {error}")
        print("Переключаюсь на процессор. Будет медленнее, но заработает.")
        self.device = "cpu"
        self.model = None
        self._load()

    def transcribe(self, audio_path: Path, speakers: int, save_words: bool) -> None:
        """
        speakers: 0 — определить число говорящих самому, 1 — не определять,
        2 и больше — столько человек в записи.
        """
        print(f"\nРаспознаю: {audio_path.name}")
        started = time.monotonic()
        audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        print(f"Длительность записи: {format_time(duration)}")

        lookup = None
        if speakers != 1:
            if self.diarizer is None:
                self.diarizer = Diarizer(MODELS_DIR)
            turns = self.diarizer.run(audio, speakers or None)
            found = len({turn.speaker for turn in turns})
            print(f"  Найдено говорящих: {found}")
            lookup = SpeakerLookup(turns)

        try:
            if self.model is None:
                self._load()
            self._run(audio_path, audio, duration, lookup, save_words)
        except RuntimeError as error:
            # Например, не хватило видеопамяти или нет нужных библиотек CUDA.
            if self.device != "cuda":
                raise
            self._switch_to_cpu(error)
            self._run(audio_path, audio, duration, lookup, save_words)

        print(f"\nГотово за {format_time(time.monotonic() - started)}.")

    def _run(self, audio_path: Path, audio, duration: float,
             lookup: SpeakerLookup | None, save_words: bool) -> None:
        print("  Перевожу речь в текст...")
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            # Время слов нужно, чтобы разрезать фразу, если в ней сменился говорящий.
            word_timestamps=lookup is not None or save_words,
        )
        if self.language is None:
            print(f"  Определён язык: {info.language}")

        base = audio_path.with_name(audio_path.stem)
        text_path = base.with_name(base.name + "_transcript.txt")
        srt_path = base.with_suffix(".srt")
        json_path = base.with_name(base.name + "_words.json")

        phrases = []
        with text_path.open("w", encoding="utf-8") as file:
            writer = TextWriter(file)
            for phrase in build_phrases(segments, lookup):
                phrases.append(phrase)
                writer.write(phrase)
                percent = min(100, phrase.end / duration * 100) if duration else 0
                who = f"{speaker_label(phrase.speaker)}: " if phrase.speaker is not None else ""
                print(f"[{percent:3.0f}%] {who}{phrase.text}")

        write_srt(srt_path, phrases)
        print(f"\nТекст: {text_path}")
        print(f"Субтитры: {srt_path}")
        if save_words:
            write_json(json_path, phrases, info.language, duration)
            print(f"Время каждого слова: {json_path}")


# ---------- пошаговый режим ----------


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def clean_dropped_path(raw: str) -> str:
    """Убирает кавычки и «& », которые Windows добавляет при перетаскивании файла в окно."""
    raw = raw.strip()
    if raw.startswith("& "):
        raw = raw[2:].strip()
    return raw.strip("\"'")


def ask_file() -> Path | None:
    while True:
        raw = ask(
            "\nПеретащите аудио- или видеофайл в это окно и нажмите Enter\n"
            "(или просто Enter, чтобы выйти): "
        )
        if not raw:
            return None
        path = Path(clean_dropped_path(raw))
        if path.is_file():
            return path
        print(f"Файл не найден: {path}. Попробуйте ещё раз.")


def print_models(recommended: str) -> None:
    print("\nДоступные модели (чем ниже, тем точнее, но медленнее и тяжелее):")
    for number, model in enumerate(MODELS, start=1):
        mark = "  <- рекомендуется" if model.name == recommended else ""
        print(f"  {number}. {model.name:<15} {model.size:>7}  {model.description}{mark}")


def ask_model(recommended: str) -> str:
    print_models(recommended)
    while True:
        raw = ask(f"\nВведите номер модели или нажмите Enter для «{recommended}»: ")
        if not raw:
            return recommended
        if raw.isdigit() and 1 <= int(raw) <= len(MODELS):
            return MODELS[int(raw) - 1].name
        if raw in MODEL_NAMES:
            return raw
        print(f"Нужно число от 1 до {len(MODELS)}.")


def ask_speakers() -> int:
    """0 — определить самому, 1 — один человек (говорящих не размечаем)."""
    while True:
        raw = ask(
            "\nСколько человек говорит в записи? Введите число\n"
            "(или просто Enter, если не знаете, — программа определит сама): "
        )
        if not raw:
            return 0
        if raw.isdigit() and int(raw) >= 1:
            return int(raw)
        print("Нужно число: 1, 2, 3 и так далее.")


def speakers_arg(value: str) -> int:
    if value == "auto":
        return 0
    if value.isdigit() and int(value) >= 1:
        return int(value)
    raise argparse.ArgumentTypeError("нужно число от 1 или auto")


# ---------- запуск ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Переводит речь из аудио- или видеофайла в текст. "
                    "Без аргументов запускается пошаговый режим с вопросами.",
    )
    parser.add_argument("files", nargs="*", type=Path, help="аудио- или видеофайлы")
    parser.add_argument(
        "-m", "--model", choices=MODEL_NAMES,
        help="модель распознавания (по умолчанию подбирается под компьютер)",
    )
    parser.add_argument(
        "-d", "--device", choices=["auto", "cuda", "cpu"], default="auto",
        help="на чём считать: видеокарта (cuda), процессор (cpu) или auto",
    )
    parser.add_argument(
        "-l", "--language", default="ru",
        help="язык записи: ru, en и т. д., или auto для автоопределения (по умолчанию ru)",
    )
    parser.add_argument(
        "-s", "--speakers", type=speakers_arg,
        help="сколько человек говорит: число, auto — определить самому, "
             "1 — не размечать говорящих (без параметра спросит или определит сам)",
    )
    parser.add_argument(
        "-w", "--words", action="store_true",
        help="дополнительно сохранить время каждого слова в <имя>_words.json",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="показать список моделей и выйти",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hardware = detect_hardware()

    if args.list_models:
        print(f"Компьютер: {hardware.describe()}")
        print_models(hardware.recommended_model())
        return 0

    print("=== Распознавание речи в текст ===")
    print(f"Компьютер: {hardware.describe()}")

    device = hardware.device if args.device == "auto" else args.device
    if device == "cuda" and not hardware.has_gpu:
        print("Видеокарта NVIDIA не найдена, поэтому работаю на процессоре.")
        device = "cpu"
    language = None if args.language == "auto" else args.language

    interactive = not args.files
    files = list(args.files)
    missing = [path for path in files if not path.is_file()]
    if missing:
        for path in missing:
            print(f"Файл не найден: {path}")
        return 1

    if interactive:
        first = ask_file()
        if first is None:
            return 0
        files = [first]

    # Модель спрашиваем, только если человек сидит у окна и не указал её сам.
    recommended = "small" if device == "cpu" else hardware.recommended_model()
    if args.model:
        model_name = args.model
    elif sys.stdin.isatty():
        model_name = ask_model(recommended)
    else:
        model_name = recommended

    # Число говорящих: из аргумента; в пошаговом режиме — для каждого файла;
    # при перетаскивании файлов на run.bat — один раз на все.
    can_ask = sys.stdin.isatty()
    batch_speakers = args.speakers
    if batch_speakers is None and not interactive:
        batch_speakers = ask_speakers() if can_ask else 0

    transcriber = Transcriber(model_name, device, language)
    failed = 0
    while files:
        path = files.pop(0)
        speakers = batch_speakers if batch_speakers is not None else ask_speakers()
        try:
            transcriber.transcribe(path, speakers, args.words)
        except Exception as error:
            failed += 1
            print(f"\nНе удалось распознать {path.name}: {error}")
        if interactive:
            next_file = ask_file()
            if next_file is not None:
                files.append(next_file)

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nОстановлено.")
        sys.exit(130)
