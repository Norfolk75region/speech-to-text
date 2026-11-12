"""Сборка фраз из результата Whisper и запись их в файлы: .txt, .srt и .json."""

import json
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
    """Кусок речи одного говорящего. speaker=None, если говорящих не определяли."""

    start: float
    end: float
    text: str
    speaker: int | None = None
    words: list[Word] = field(default_factory=list)


def speaker_label(speaker: int | None) -> str:
    return f"Спикер {speaker}" if speaker is not None else ""


def build_phrases(segments: Iterable, lookup: SpeakerLookup | None) -> Iterator[Phrase]:
    """
    Превращает сегменты Whisper во фразы.

    Если есть время слов, сегмент режется там, где меняется говорящий:
    Whisper не знает о голосах и легко склеивает реплики двух людей в одну строку.
    """
    for segment in segments:
        if not segment.text.strip():
            continue
        if not segment.words:
            speaker = lookup.speaker_at(segment.start, segment.end) if lookup else None
            yield Phrase(segment.start, segment.end, segment.text.strip(), speaker)
            continue

        current: Phrase | None = None
        for raw in segment.words:
            word = Word(raw.start, raw.end, raw.word, raw.probability)
            speaker = lookup.speaker_at(word.start, word.end) if lookup else None
            if current is not None and speaker != current.speaker:
                current.text = current.text.strip()
                if current.text:
                    yield current
                current = None
            if current is None:
                current = Phrase(word.start, word.end, "", speaker)
            current.words.append(word)
            current.text += word.word
            current.end = word.end
        if current is not None:
            current.text = current.text.strip()
            if current.text:
                yield current


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
        for number, phrase in enumerate(phrases, start=1):
            prefix = f"{speaker_label(phrase.speaker)}: " if phrase.speaker is not None else ""
            file.write(f"{number}\n{srt_clock(phrase.start)} --> {srt_clock(phrase.end)}\n")
            file.write(f"{prefix}{phrase.text}\n\n")


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
