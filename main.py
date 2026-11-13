"""
Распознавание речи из аудио- и видеофайлов в текст (faster-whisper).

Без аргументов открывается меню: в окно можно перетащить файл, несколько
файлов или папку, а ещё зайти в настройки (модель и язык по умолчанию,
хранятся в settings.json) или запустить с другими параметрами на один раз.
С файлами или папками в аргументах распознаёт их сразу:

    python main.py запись.m4a
    python main.py интервью.mp3 --speakers 2 --words
    python main.py лекция.mp4 --model small --device cpu --speakers 1
    python main.py папка_с_записями
    python main.py --list-models

Рядом с записью появляются <имя>_transcript.txt (текст с временем и говорящими),
<имя>.srt (субтитры) и с --words <имя>_words.json (время каждого слова).
С --output результаты пишутся в указанную папку, с --srt-only — только субтитры.
"""

import argparse
import os
import re
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
from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio, download_model

from chunking import plan_chunks
from diarization import SAMPLE_RATE, Diarizer, SpeakerLookup
from errors import UserError, explain, is_out_of_memory
from formats import TextWriter, build_phrases, drop_invented, speaker_label, write_json, write_srt
from settings import Settings, load_settings, save_settings

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


def is_model_downloaded(name: str) -> bool:
    """Лежит ли модель в models/. Смотрим на сам файл весов: папка остаётся и от прерванной загрузки."""
    pattern = f"models--*--faster-whisper-{name}/snapshots/*/model.bin"
    return any(MODELS_DIR.glob(pattern))


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


# Образец текста, который модель видит перед каждым куском записи. Смысл не важен,
# важен вид: предложения с заглавной буквы, точки, запятые. Для остальных языков
# подсказки нет — модель пишет как умеет.
STYLE_PROMPTS = {
    "ru": "Здравствуйте. Итак, начнём: это запись разговора, с правильной пунктуацией и заглавными буквами.",
    "en": "Hello. Let's begin: this is a recording, with proper punctuation and capital letters.",
}

# Сколько кусков записи распознавать за один проход. На видеокарте так заметно
# быстрее; большая модель с четырьмя кусками помещается в 4 ГБ видеопамяти.
BATCH_SIZE = {"cuda": 4, "cpu": 2}


@dataclass(frozen=True)
class Options:
    """Что делать с записями, кроме модели и языка."""

    speakers: int = 0                # 0 — определить самому, 1 — не определять, иначе число людей
    save_words: bool = False         # писать <имя>_words.json
    output_dir: Path | None = None   # куда класть результаты; None — рядом с записью
    srt_only: bool = False           # только субтитры, без <имя>_transcript.txt


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
        if not is_model_downloaded(self.model_name):
            print("Модель скачивается из интернета в первый раз, это может занять несколько минут.")
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=compute_type,
            download_root=str(MODELS_DIR),
        )

    def _switch_to_cpu(self, error: Exception) -> None:
        if is_out_of_memory(error):
            print("\nВидеокарте не хватило памяти для этой модели.")
        else:
            print(f"\nНа видеокарте не получилось (причина: {error}).")
        print("Переключаюсь на процессор. Будет медленнее, но заработает.")
        self.device = "cpu"
        self.model = None
        self._load()

    @staticmethod
    def _read_audio(audio_path: Path):
        """Достаёт звук из файла. Ошибки чтения объясняет простыми словами."""
        no_sound = UserError(
            "В этом файле нет звука: похоже, это видео без звуковой дорожки "
            "или пустая запись."
        )
        try:
            audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
        except IndexError:
            # Так декодер сообщает, что в файле нет звуковой дорожки.
            raise no_sound from None
        except (OSError, MemoryError):
            raise  # нет доступа, файл пропал, мало памяти — объяснит explain()
        except Exception as error:
            raise UserError(
                "Не получилось прочитать файл: это не аудио и не видео, либо он "
                "повреждён или скопирован не до конца."
            ) from error
        if len(audio) == 0:
            raise no_sound
        return audio

    def transcribe(self, audio_path: Path, options: Options) -> None:
        print(f"\nРаспознаю: {audio_path.name}")
        started = time.monotonic()
        audio = self._read_audio(audio_path)
        duration = len(audio) / SAMPLE_RATE
        print(f"Длительность записи: {format_time(duration)}")

        lookup = None
        if options.speakers != 1:
            if self.diarizer is None:
                self.diarizer = Diarizer(MODELS_DIR)
            turns = self.diarizer.run(audio, options.speakers or None)
            found = len({turn.speaker for turn in turns})
            print(f"  Найдено говорящих: {found}")
            lookup = SpeakerLookup(turns)

        try:
            if self.model is None:
                self._load()
            self._run(audio_path, audio, duration, lookup, options)
        except RuntimeError as error:
            # Например, не хватило видеопамяти или нет нужных библиотек CUDA.
            if self.device != "cuda":
                raise
            self._switch_to_cpu(error)
            self._run(audio_path, audio, duration, lookup, options)

        print(f"\nГотово за {format_time(time.monotonic() - started)}.")

    def _run(self, audio_path: Path, audio, duration: float,
             lookup: SpeakerLookup | None, options: Options) -> None:
        print("  Перевожу речь в текст...")
        # Запись режем на куски по паузам и распознаём их независимо: если идти
        # по длинной записи подряд, модель начинает повторять фразы и теряет пунктуацию.
        chunks = plan_chunks(audio)
        if not chunks:
            raise UserError("В этой записи не слышно речи: только тишина, шум или музыка.")

        language = self.language
        if language is None:
            # Язык определяем по первому куску с речью, а не по началу файла: там бывает музыка.
            first = audio[int(chunks[0].start * SAMPLE_RATE):int(chunks[0].end * SAMPLE_RATE)]
            language = self.model.detect_language(first)[0]
            print(f"  Определён язык: {language}")

        segments, info = BatchedInferencePipeline(self.model).transcribe(
            audio,
            language=language,
            beam_size=5,
            batch_size=BATCH_SIZE[self.device],
            vad_filter=False,
            clip_timestamps=[{"start": chunk.start, "end": chunk.end} for chunk in chunks],
            # Подсказка задаёт стиль каждому куску: без неё модель через раз
            # пишет сплошной текст без точек и заглавных букв.
            initial_prompt=STYLE_PROMPTS.get(language),
            # Время слов нужно всегда: по нему собираются предложения и субтитры,
            # а при разметке говорящих — режутся реплики.
            word_timestamps=True,
        )

        # Имена собираем из stem, а не через with_suffix: в названии записи
        # бывают точки («Встреча 12.11.2026»), и with_suffix отрезал бы хвост.
        folder = options.output_dir or audio_path.parent
        folder.mkdir(parents=True, exist_ok=True)
        stem = audio_path.stem
        text_path = folder / (stem + "_transcript.txt")
        srt_path = folder / (stem + ".srt")
        json_path = folder / (stem + "_words.json")

        phrases = []
        dropped = []
        file = None if options.srt_only else text_path.open("w", encoding="utf-8")
        try:
            writer = TextWriter(file) if file else None
            for phrase in drop_invented(build_phrases(segments, lookup), dropped):
                phrases.append(phrase)
                if writer:
                    writer.write(phrase)
                percent = min(100, phrase.end / duration * 100) if duration else 0
                who = f"{speaker_label(phrase.speaker)}: " if phrase.speaker is not None else ""
                print(f"[{percent:3.0f}%] {who}{phrase.text}")
        finally:
            if file:
                file.close()

        write_srt(srt_path, phrases)
        print()
        for phrase in dropped:
            print("Убрана фраза, которую модель, скорее всего, придумала сама: "
                  f"[{format_time(phrase.start)}] {phrase.text}")
        if not options.srt_only:
            print(f"Текст: {text_path}")
        print(f"Субтитры: {srt_path}")
        if options.save_words:
            write_json(json_path, phrases, info.language, duration)
            print(f"Время каждого слова: {json_path}")


# ---------- файлы ----------

# Что брать из папки. Файл, указанный напрямую, берём с любым расширением.
MEDIA_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".flac", ".wma", ".amr", ".aiff",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".m4v", ".3gp", ".mpg", ".mpeg", ".ts", ".flv",
}


def split_dropped(raw: str) -> list[str]:
    """
    Разбирает строку, которая получается при перетаскивании в окно: один путь
    или несколько через пробел. Пути с пробелами Windows берёт в кавычки,
    а PowerShell ещё и ставит «& » в начале.
    """
    raw = raw.strip()
    if raw.startswith("& "):
        raw = raw[2:].strip()
    if not raw:
        return []
    # Один путь, возможно с пробелами и без кавычек (набрали руками).
    whole = raw.strip("\"'")
    if Path(whole).exists():
        return [whole]
    parts = re.findall(r'"([^"]+)"|\'([^\']+)\'|(\S+)', raw)
    paths = [next(filter(None, groups)) for groups in parts]
    # Ни одного куска не существует — значит, это один неверный путь с пробелами,
    # а не несколько файлов: не дробим его, чтобы не пугать списком обрывков.
    if not any(Path(path).exists() for path in paths):
        return [whole]
    return paths


def expand_paths(paths: list) -> tuple[list[Path], list[str]]:
    """
    Превращает файлы и папки в список файлов для распознавания.
    Из папки берутся аудио и видео, вложенные папки не просматриваются.
    Второе значение — что не так, простыми словами.
    """
    files: list[Path] = []
    problems: list[str] = []
    for path in map(Path, paths):
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            inside = sorted(
                item for item in path.iterdir()
                if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS
            )
            if inside:
                files.extend(inside)
            else:
                problems.append(f"В папке нет аудио и видео: {path}")
        else:
            problems.append(f"Файл не найден: {path}")
    return list(dict.fromkeys(files)), problems


def read_files(raw: str) -> list[Path]:
    """Файлы из строки, введённой в окне. Пустой список — что-то не нашлось, об этом уже сказано."""
    files, problems = expand_paths(split_dropped(raw))
    if problems:
        for problem in problems:
            print(problem)
        print("Попробуйте ещё раз.")
        return []
    if len(files) > 1:
        print(f"Файлов к распознаванию: {len(files)}")
    return files


# ---------- вопросы ----------

LANGUAGES = {"ru": "русский", "en": "английский", "auto": "определять автоматически"}
YES = {"д", "да", "y", "yes"}
NO = {"н", "нет", "n", "no"}


def language_title(code: str) -> str:
    return LANGUAGES.get(code, code)


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def ask_yes(prompt: str, default: bool) -> bool:
    while True:
        raw = ask(prompt).lower()
        if not raw:
            return default
        if raw in YES:
            return True
        if raw in NO:
            return False
        print("Введите «да» или «нет».")


def print_models(recommended: str, current: str | None = None) -> None:
    print("\nДоступные модели (чем ниже, тем точнее, но медленнее и тяжелее):")
    for number, model in enumerate(MODELS, start=1):
        status = "скачана" if is_model_downloaded(model.name) else "не скачана"
        marks = []
        if model.name == current:
            marks.append("выбрана")
        if model.name == recommended:
            marks.append("рекомендуется")
        mark = f"  <- {', '.join(marks)}" if marks else ""
        print(f"  {number}. {model.name:<15} {model.size:>7}  {status:<10}  {model.description}{mark}")


def ask_model(recommended: str, current: str) -> str:
    print_models(recommended, current)
    while True:
        raw = ask(f"\nВведите номер модели или нажмите Enter для «{current}»: ")
        if not raw:
            return current
        if raw.isdigit() and 1 <= int(raw) <= len(MODELS):
            return MODELS[int(raw) - 1].name
        if raw in MODEL_NAMES:
            return raw
        print(f"Нужно число от 1 до {len(MODELS)}.")


def ask_language(current: str) -> str:
    codes = list(LANGUAGES)
    print("\nЯзык записи:")
    for number, code in enumerate(codes, start=1):
        print(f"  {number}. {LANGUAGES[code]}")
    while True:
        raw = ask(f"\nВведите номер или нажмите Enter для «{language_title(current)}»: ")
        if not raw:
            return current
        if raw.isdigit() and 1 <= int(raw) <= len(codes):
            return codes[int(raw) - 1]
        print(f"Нужно число от 1 до {len(codes)}.")


def ask_speakers(file_count: int = 1) -> int:
    """0 — определить самому, 1 — один человек (говорящих не размечаем)."""
    where = "в записи" if file_count == 1 else "в записях (одно число на все файлы)"
    while True:
        raw = ask(
            f"\nСколько человек говорит {where}? Введите число\n"
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


# ---------- настройки ----------


def describe_choice(model_name: str, language: str) -> str:
    status = "скачана" if is_model_downloaded(model_name) else "скачается при первом запуске"
    return f"модель {model_name} ({status}), язык: {language_title(language)}"


def download_everything(model_name: str) -> None:
    """Скачивает модель распознавания и модели для определения говорящих."""
    try:
        if not is_model_downloaded(model_name):
            size = next(model.size for model in MODELS if model.name == model_name)
            print(f"\nСкачиваю модель {model_name} ({size}), это может занять несколько минут...")
            download_model(model_name, cache_dir=str(MODELS_DIR))
        Diarizer(MODELS_DIR).download_all()
    except Exception as error:
        print("\nСкачать не получилось.")
        print(explain(error))
        return
    print("\nВсё нужное скачано: дальше программа работает без интернета.")


def store(settings: Settings) -> None:
    if not save_settings(settings):
        print(
            "Не получилось сохранить настройки: в папку с программой нельзя записывать. "
            "Они действуют, пока открыто это окно."
        )


def settings_menu(settings: Settings, hardware: Hardware, recommended: str) -> None:
    while True:
        model_name = settings.model or recommended
        ready = is_model_downloaded(model_name) and Diarizer(MODELS_DIR).is_downloaded()
        print("\n--- Настройки по умолчанию ---")
        print(f"Компьютер: {hardware.describe()}")
        print(f"  1. Модель: {model_name} ({'скачана' if is_model_downloaded(model_name) else 'не скачана'})")
        print(f"  2. Язык записи: {language_title(settings.language)}")
        print(f"  3. Скачать всё для работы без интернета{' (уже скачано)' if ready else ''}")
        raw = ask("\nВведите номер или нажмите Enter, чтобы вернуться: ")
        if not raw:
            return
        if raw == "1":
            settings.model = ask_model(recommended, model_name)
            store(settings)
            if not is_model_downloaded(settings.model) and ask_yes(
                "Эта модель ещё не скачана. Скачать сейчас? (Enter — да, «нет» — позже): ", True
            ):
                download_everything(settings.model)
        elif raw == "2":
            settings.language = ask_language(settings.language)
            store(settings)
        elif raw == "3":
            download_everything(model_name)
        else:
            print("Нужно число от 1 до 3.")


# ---------- запуск ----------


class Session:
    """Распознаёт пачки файлов и держит загруженную модель между ними."""

    def __init__(self, device: str):
        self.device = device
        self.transcriber: Transcriber | None = None

    def run(self, files: list[Path], model_name: str, language: str, options: Options) -> int:
        """Возвращает, сколько файлов распознать не удалось."""
        if self.transcriber is None or self.transcriber.model_name != model_name:
            self.transcriber = Transcriber(model_name, self.device, None)
        self.transcriber.language = None if language == "auto" else language

        failed = 0
        for path in files:
            try:
                self.transcriber.transcribe(path, options)
            except Exception as error:
                failed += 1
                print(f"\nНе удалось распознать {path.name}.")
                print(explain(error))
        if len(files) > 1:
            print(f"\nРаспознано файлов: {len(files) - failed} из {len(files)}.")
        return failed


def main_menu(session: Session, settings: Settings, hardware: Hardware,
              recommended: str, args: argparse.Namespace) -> int:
    """Главный экран: перетащили файл — распознаём с настройками по умолчанию."""
    failed = 0
    while True:
        # Параметры из командной строки важнее сохранённых настроек.
        model_name = args.model or settings.model or recommended
        language = args.language or settings.language
        save_words = args.words

        print(f"\nСейчас: {describe_choice(model_name, language)}")
        raw = ask(
            "\nПеретащите в это окно файл, несколько файлов или папку и нажмите Enter.\n"
            "Или введите: 1 — настройки, 2 — запуск с другими параметрами, Enter — выход\n"
            "> "
        )
        if not raw:
            return failed
        if raw == "1":
            settings_menu(settings, hardware, recommended)
            continue
        if raw == "2":
            raw = ask(
                "\nПеретащите файл, несколько файлов или папку и нажмите Enter\n"
                "(или просто Enter, чтобы вернуться): "
            )
            files = read_files(raw) if raw else []
            if not files:
                continue
            print("\nЭти параметры действуют один раз, настройки по умолчанию не меняются.")
            model_name = ask_model(recommended, model_name)
            language = ask_language(language)
            save_words = ask_yes(
                "\nСохранить время каждого слова в отдельный файл? (Enter — нет, «да» — сохранить): ",
                False,
            )
        else:
            files = read_files(raw)
            if not files:
                continue

        speakers = args.speakers if args.speakers is not None else ask_speakers(len(files))
        options = Options(speakers, save_words, args.output, args.srt_only)
        failed += session.run(files, model_name, language, options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Переводит речь из аудио- или видеофайла в текст. "
                    "Без аргументов открывается меню с настройками.",
    )
    parser.add_argument(
        "files", nargs="*", type=Path,
        help="аудио- или видеофайлы; из папки берутся все записи",
    )
    parser.add_argument(
        "-m", "--model", choices=MODEL_NAMES,
        help="модель распознавания (по умолчанию из настроек или под компьютер)",
    )
    parser.add_argument(
        "-d", "--device", choices=["auto", "cuda", "cpu"], default="auto",
        help="на чём считать: видеокарта (cuda), процессор (cpu) или auto",
    )
    parser.add_argument(
        "-l", "--language",
        help="язык записи: ru, en и т. д., или auto для автоопределения "
             "(по умолчанию из настроек, изначально ru)",
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
        "-o", "--output", type=Path, metavar="ПАПКА",
        help="папка для результатов (по умолчанию они ложатся рядом с записью)",
    )
    parser.add_argument(
        "--srt-only", action="store_true",
        help="писать только субтитры, без <имя>_transcript.txt",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="показать список моделей и выйти",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hardware = detect_hardware()
    settings = load_settings(MODEL_NAMES)

    if args.list_models:
        print(f"Компьютер: {hardware.describe()}")
        print_models(hardware.recommended_model(), settings.model)
        return 0

    print("=== Распознавание речи в текст ===")
    print(f"Компьютер: {hardware.describe()}")

    device = hardware.device if args.device == "auto" else args.device
    if device == "cuda" and not hardware.has_gpu:
        print("Видеокарта NVIDIA не найдена, поэтому работаю на процессоре.")
        device = "cpu"
    recommended = "small" if device == "cpu" else hardware.recommended_model()
    session = Session(device)

    if not args.files:
        return 1 if main_menu(session, settings, hardware, recommended, args) else 0

    # Файлы переданы сразу (например, их перетащили на run.bat): меню не показываем,
    # берём настройки по умолчанию и спрашиваем только число говорящих.
    files, problems = expand_paths(args.files)
    if problems:
        for problem in problems:
            print(problem)
        return 1
    model_name = args.model or settings.model or recommended
    language = args.language or settings.language
    print(f"Сейчас: {describe_choice(model_name, language)}")
    if len(files) > 1:
        print(f"Файлов к распознаванию: {len(files)}")

    speakers = args.speakers
    if speakers is None:
        speakers = ask_speakers(len(files)) if sys.stdin.isatty() else 0
    options = Options(speakers, args.words, args.output, args.srt_only)
    return 1 if session.run(files, model_name, language, options) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nОстановлено.")
        sys.exit(130)
