"""Сборка фраз из результата Whisper и запись их в файлы: .txt, .srt и .json."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, TextIO

from diarization import SpeakerLookup


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class Phrase:
    """Предложение одного говорящего. speaker=None, если говорящих не определяли."""

    start: float
    end: float
    text: str
    speaker: int | None = None
    words: list[Word] = field(default_factory=list)


@dataclass
class Cue:
    """Одна реплика субтитров."""

    start: float
    end: float
    text: str
    speaker: int | None = None


def speaker_label(speaker: int | None) -> str:
    return f"Спикер {speaker}" if speaker is not None else ""


# ---------- фразы ----------

# Фраза в тексте — одно предложение. Если модель не расставила точки,
# режем по длине, иначе вся запись окажется одной строкой.
PHRASE_MAX_SECONDS = 30.0
PHRASE_MAX_CHARS = 300
# Кусок предложения короче этого не считается отдельной репликой другого человека.
RUN_MIN_WORDS = 2
RUN_MIN_SECONDS = 1.0

# После этих сокращений точка не означает конец предложения.
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "vs", "etc", "prof", "sr", "jr", "no",
    "г", "гг", "ул", "им", "руб", "коп", "тыс", "млн", "млрд", "др", "пр", "см", "стр", "рис",
}
CLOSING = "\"'”»)]"


def ends_sentence(text: str) -> bool:
    """Заканчивается ли слово концом предложения (а не сокращением вроде «г.» или «т. е.»)."""
    text = text.strip().rstrip(CLOSING)
    if not text:
        return False
    if text[-1] in "?!…" or text.endswith(".."):
        return True
    if text[-1] != ".":
        return False
    core = text[:-1].lstrip("\"'“«([")
    # Одна буква с точкой — инициал или кусок «т. е.», «т. д.».
    if len(core) == 1 and core.isalpha():
        return False
    return core.lower() not in ABBREVIATIONS


def join_words(words: list[Word]) -> str:
    # Whisper отдаёт слова уже с пробелом впереди там, где он нужен.
    return "".join(word.word for word in words).strip()


def build_phrases(segments: Iterable, lookup: SpeakerLookup | None) -> Iterator[Phrase]:
    """
    Превращает сегменты Whisper во фразы.

    Сегменты Whisper режутся по времени, а не по смыслу, и часто обрываются
    посреди предложения. Поэтому фразы собираем заново из слов: фраза кончается
    там, где кончается предложение или меняется говорящий (Whisper о голосах
    не знает и легко склеивает реплики двух людей в одну строку).
    """
    sentence: list[Word] = []
    speakers: list[int | None] = []

    def flush() -> Iterator[Phrase]:
        nonlocal sentence, speakers
        for speaker, words in _speaker_runs(sentence, speakers):
            text = join_words(words)
            if text:
                yield Phrase(words[0].start, words[-1].end, text, speaker, words)
        sentence, speakers = [], []

    for segment in segments:
        if not segment.text.strip():
            continue
        if not segment.words:
            # Времени слов нет — берём сегмент как есть.
            yield from flush()
            speaker = lookup.speaker_at(segment.start, segment.end) if lookup else None
            yield Phrase(segment.start, segment.end, segment.text.strip(), speaker)
            continue

        for raw in segment.words:
            word = Word(raw.start, raw.end, raw.word, raw.probability)
            if sentence and (
                word.end - sentence[0].start > PHRASE_MAX_SECONDS
                or len(join_words(sentence)) + len(word.word) > PHRASE_MAX_CHARS
            ):
                yield from flush()
            sentence.append(word)
            speakers.append(lookup.speaker_at(word.start, word.end) if lookup else None)
            if ends_sentence(word.word):
                yield from flush()
    yield from flush()


def _speaker_runs(words: list[Word], speakers: list) -> list[tuple[int | None, list[Word]]]:
    """
    Делит предложение на куски по говорящим.

    Граница между голосами известна неточно, и на стыке одно-два слова часто
    достаются соседу. Такой обрывок посреди предложения отдельной репликой
    не считаем — приклеиваем к соседнему куску.
    """
    runs: list[tuple[int | None, list[Word]]] = []
    for word, speaker in zip(words, speakers):
        if runs and runs[-1][0] == speaker:
            runs[-1][1].append(word)
        else:
            runs.append((speaker, [word]))

    index = 0
    while len(runs) > 1 and index < len(runs):
        speaker, run = runs[index]
        if len(run) <= RUN_MIN_WORDS and run[-1].end - run[0].start < RUN_MIN_SECONDS:
            # Обрывок в начале предложения относится к тому, кто говорит дальше.
            target = index - 1 if index > 0 else 1
            merged = runs[target][1] + run if target < index else run + runs[target][1]
            runs[target] = (runs[target][0], merged)
            del runs[index]
            # После склейки соседи могли оказаться одним говорящим.
            joined: list[tuple[int | None, list[Word]]] = []
            for who, part in runs:
                if joined and joined[-1][0] == who:
                    joined[-1] = (who, joined[-1][1] + part)
                else:
                    joined.append((who, part))
            runs = joined
            index = 0
        else:
            index += 1
    return runs


# ---------- выдумки модели ----------

# На тишине, музыке и в самом конце записи Whisper дописывает фразы, которых
# никто не говорил: он видел их в субтитрах, на которых учился.
# Подписи авторов субтитров настоящей речью не бывают — выбрасываем всегда.
INVENTED_ALWAYS = [
    r"субтитры (сделал|создавал|делал|подготовил|подогнал|создал)\b.*",
    r"редактор субтитров\b.*",
    r"корректор [а-я] ?[а-я]+",
    r"(subtitles|captions|transcribed|transcription) by\b.*",
    r"subtitles (made|created|provided) by\b.*",
    r".*\bamara org\b.*",
]
# А такое человек мог сказать и сам. Выбрасываем, только если фраза стоит
# особняком (тишина или музыка вокруг) либо модель сама в ней не уверена.
INVENTED_MAYBE = [
    r"(thank you|thanks)( so much| very much)? for watching.*",
    r"(please )?(like and )?subscribe( to (my|our|the) channel)?",
    r"see you (in the )?next (time|video)",
    r"продолжение следует",
    r"спасибо за просмотр",
    r"(спасибо за внимание )?до новых встреч",
    r"подписывайтесь на (мой |наш )?канал",
]
INVENTED_PAUSE = 1.5        # тишина с обеих сторон, секунды
INVENTED_PROBABILITY = 0.6  # средняя уверенность модели в словах фразы


def _plain(text: str) -> str:
    """Текст без знаков и регистра — для сравнения."""
    text = text.lower().replace("ё", "е")
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def _matches(text: str, patterns: list[str]) -> bool:
    return any(re.fullmatch(pattern, text) for pattern in patterns)


def drop_invented(phrases: Iterable[Phrase], dropped: list[Phrase] | None = None) -> Iterator[Phrase]:
    """
    Пропускает фразы дальше, выбрасывая типовые выдумки модели.
    Выброшенное складывает в dropped, чтобы о нём можно было сказать.
    """
    previous_end = 0.0
    held: Phrase | None = None  # кандидат на выброс: ждём, какая пауза будет после него

    def is_alone(phrase: Phrase, before: float, after: float) -> bool:
        if phrase.words:
            sure = sum(word.probability for word in phrase.words) / len(phrase.words)
            if sure < INVENTED_PROBABILITY:
                return True
        return before >= INVENTED_PAUSE and after >= INVENTED_PAUSE

    def reject(phrase: Phrase) -> None:
        if dropped is not None:
            dropped.append(phrase)

    for phrase in phrases:
        if held is not None:
            if is_alone(held, held.start - previous_end, phrase.start - held.end):
                reject(held)
            else:
                yield held
                previous_end = held.end
            held = None

        plain = _plain(phrase.text)
        if _matches(plain, INVENTED_ALWAYS):
            reject(phrase)
        elif _matches(plain, INVENTED_MAYBE):
            held = phrase
        else:
            yield phrase
            previous_end = phrase.end

    if held is not None:
        # Последняя фраза записи: после неё тишина по определению.
        if is_alone(held, held.start - previous_end, INVENTED_PAUSE):
            reject(held)
        else:
            yield held


# ---------- субтитры ----------

CUE_LINE_CHARS = 42      # знаков в строке; в реплике не больше двух строк
CUE_MAX_SECONDS = 7.0
CUE_MIN_SECONDS = 1.0    # короче зритель не успеет прочитать
CUE_PAUSE = 2.0          # пауза в речи длиннее этой разрывает реплику
CUE_GAP = 0.04           # зазор между соседними репликами
CUE_MERGE_GAP = 0.6      # короткие предложения подряд склеиваем в одну реплику


def _cue_prefix(speaker: int | None) -> str:
    return f"{speaker_label(speaker)}: " if speaker is not None else ""


def _fits(words: list[Word], prefix: str) -> bool:
    """Помещается ли реплика в две строки и в отведённое время."""
    if words[-1].end - words[0].start > CUE_MAX_SECONDS:
        return False
    lines = wrap_cue(prefix + join_words(words)).split("\n")
    return all(len(line) <= CUE_LINE_CHARS for line in lines)


def _split_for_cues(words: list[Word], prefix: str) -> list[list[Word]]:
    """Делит слова одной фразы на реплики: по паузам, запятым, а если иначе никак — по длине."""
    # Сначала на части, внутри которых резать нежелательно.
    units: list[list[Word]] = []
    after_pause: list[bool] = []  # перед этой частью пауза: там режем обязательно
    new_unit = True
    for index, word in enumerate(words):
        pause = index > 0 and word.start - words[index - 1].end > CUE_PAUSE
        if new_unit or pause:
            units.append([])
            after_pause.append(pause)
        units[-1].append(word)
        new_unit = word.word.strip().rstrip(CLOSING)[-1:] in ",;:—–"

    groups: list[list[Word]] = []
    current: list[Word] = []
    for unit, pause in zip(units, after_pause):
        if current and (pause or not _fits(current + unit, prefix)):
            groups.append(current)
            current = []
        current = current + unit
        # Одна часть без запятых длиннее предела — режем по словам.
        while not _fits(current, prefix) and len(current) > 1:
            longest = len(current) - 1
            while longest > 1 and not _fits(current[:longest], prefix):
                longest -= 1
            # Из подходящих мест выбираем то, где говорящий сделал самую заметную паузу.
            size = max(
                range(max(1, longest // 2), longest + 1),
                key=lambda count: (round(current[count].start - current[count - 1].end, 1), count),
            )
            groups.append(current[:size])
            current = current[size:]
    if current:
        groups.append(current)
    return groups


def build_cues(phrases: list[Phrase]) -> list[Cue]:
    """
    Собирает реплики субтитров: не длиннее 7 секунд и двух строк по 42 знака.
    Длинные предложения режутся по запятым, короткие подряд — склеиваются.
    Реплики не накладываются друг на друга, короткая держится на экране не меньше секунды.
    """
    cues: list[Cue] = []
    for phrase in phrases:
        prefix = _cue_prefix(phrase.speaker)
        if not phrase.words:
            cues.append(Cue(phrase.start, phrase.end, phrase.text, phrase.speaker))
            continue
        for group in _split_for_cues(phrase.words, prefix):
            cue = Cue(group[0].start, group[-1].end, join_words(group), phrase.speaker)
            last = cues[-1] if cues else None
            if (
                last is not None
                and last.speaker == cue.speaker
                and cue.start - last.end <= CUE_MERGE_GAP
                and len(prefix) + len(last.text) + 1 + len(cue.text) <= CUE_LINE_CHARS
                and cue.end - last.start <= CUE_MAX_SECONDS
                and ends_sentence(last.text)
            ):
                last.text = f"{last.text} {cue.text}"
                last.end = cue.end
            else:
                cues.append(cue)

    # Время: по возрастанию, без наложений, короткие реплики держим подольше.
    cues.sort(key=lambda cue: cue.start)
    for index, cue in enumerate(cues):
        if index:
            cue.start = max(cue.start, cues[index - 1].end)
        cue.end = max(cue.end, cue.start + CUE_MIN_SECONDS)
        if index + 1 < len(cues):
            limit = max(cues[index + 1].start - CUE_GAP, cue.start + 0.1)
            cue.end = min(cue.end, limit)
    return cues


def wrap_cue(text: str, width: int = CUE_LINE_CHARS) -> str:
    """Делит длинную реплику на две строки примерно пополам."""
    if len(text) <= width or " " not in text:
        return text
    spaces = [index for index, char in enumerate(text) if char == " "]
    middle = min(spaces, key=lambda index: abs(index - len(text) / 2))
    return text[:middle] + "\n" + text[middle + 1:]


# ---------- запись в файлы ----------


def clock(seconds: float) -> str:
    """00:01:23 — для текста."""
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def srt_clock(seconds: float) -> str:
    """00:01:23,456 — формат времени в субтитрах."""
    millis = round(seconds * 1000)
    return f"{millis // 3_600_000:02d}:{millis % 3_600_000 // 60_000:02d}:{millis % 60_000 // 1000:02d},{millis % 1000:03d}"


class TextWriter:
    """
    Пишет .txt по мере распознавания, чтобы текст не пропал, если программу закроют.

    С говорящими реплики группируются:
        Спикер 1
        [00:00:00] Первая фраза.
        [00:00:04] Вторая фраза.

        Спикер 2
        [00:00:09] Ответ.
    """

    def __init__(self, file: TextIO):
        self.file = file
        self.last_speaker: int | None = None
        self.first = True

    def write(self, phrase: Phrase) -> None:
        if phrase.speaker is not None and (self.first or phrase.speaker != self.last_speaker):
            if not self.first:
                self.file.write("\n")
            self.file.write(speaker_label(phrase.speaker) + "\n")
            self.last_speaker = phrase.speaker
        self.file.write(f"[{clock(phrase.start)}] {phrase.text}\n")
        self.file.flush()
        self.first = False


def write_srt(path: Path, phrases: list[Phrase]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for number, cue in enumerate(build_cues(phrases), start=1):
            file.write(f"{number}\n{srt_clock(cue.start)} --> {srt_clock(cue.end)}\n")
            file.write(f"{wrap_cue(_cue_prefix(cue.speaker) + cue.text)}\n\n")


def write_json(path: Path, phrases: list[Phrase], language: str, duration: float) -> None:
    speakers = sorted({phrase.speaker for phrase in phrases if phrase.speaker is not None})
    data = {
        "language": language,
        "duration": round(duration, 2),
        "speakers": len(speakers),
        "phrases": [
            {
                "start": round(phrase.start, 2),
                "end": round(phrase.end, 2),
                "speaker": phrase.speaker,
                "text": phrase.text,
                "words": [
                    {
                        "start": round(word.start, 2),
                        "end": round(word.end, 2),
                        "word": word.word.strip(),
                        "probability": round(word.probability, 3),
                    }
                    for word in phrase.words
                ],
            }
            for phrase in phrases
        ],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
