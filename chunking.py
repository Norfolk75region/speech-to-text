"""
Нарезка записи на куски по паузам в речи.

Whisper слушает запись окнами по 30 секунд. Если идти по длинной записи подряд
и подсказывать модели её же предыдущий текст, ошибка в одном окне тянется
дальше: фразы повторяются, пропадает пунктуация, текст превращается в кашу.
Поэтому запись режется на независимые куски не длиннее окна, а границы
ставятся в самых длинных паузах — там, где говорящий закончил мысль.
"""

from dataclasses import dataclass

from faster_whisper.vad import VadOptions, get_speech_timestamps

from diarization import SAMPLE_RATE

MAX_CHUNK = 29.0   # кусок должен целиком помещаться в окно модели (30 с)
MIN_CHUNK = 12.0   # короче не режем без нужды: модели нужен контекст
LONG_PAUSE = 3.0   # тишину и музыку длиннее этого модели не показываем: на них она выдумывает текст
PADDING = 0.2      # запас по краям куска, чтобы не срезать начало и конец слова


@dataclass(frozen=True)
class Chunk:
    start: float
    end: float


def plan_chunks(audio) -> list[Chunk]:
    """Возвращает куски записи с речью, в секундах. Пустой список — речи в записи нет."""
    speeches = get_speech_timestamps(
        audio,
        VadOptions(
            min_silence_duration_ms=200,
            speech_pad_ms=0,
            # Речь без единой паузы дольше окна детектор порежет сам.
            max_speech_duration_s=MAX_CHUNK - 2 * PADDING,
        ),
    )
    spans = [(speech["start"] / SAMPLE_RATE, speech["end"] / SAMPLE_RATE) for speech in speeches]
    total = len(audio) / SAMPLE_RATE

    chunks: list[Chunk] = []
    first = 0
    while first < len(spans):
        begin = spans[first][0]
        # Докуда можно дотянуть кусок: пока помещается в окно и нет долгой тишины.
        last = first
        while (
            last + 1 < len(spans)
            and spans[last + 1][1] - begin <= MAX_CHUNK - 2 * PADDING
            and spans[last + 1][0] - spans[last][1] <= LONG_PAUSE
        ):
            last += 1
        stopped_by_length = (
            last + 1 < len(spans) and spans[last + 1][0] - spans[last][1] <= LONG_PAUSE
        )
        if stopped_by_length:
            # Упёрлись в длину окна — режем в самой длинной паузе из подходящих.
            candidates = [
                index for index in range(first, last + 1)
                if spans[index][1] - begin >= MIN_CHUNK
            ] or [last]
            last = max(
                candidates,
                key=lambda index: (round(spans[index + 1][0] - spans[index][1], 2), index),
            )

        left = max(0.0, begin - PADDING)
        right = min(total, spans[last][1] + PADDING)
        if chunks:
            left = max(left, chunks[-1].end)
        if last + 1 < len(spans):
            right = min(right, (spans[last][1] + spans[last + 1][0]) / 2)
        chunks.append(Chunk(left, right))
        first = last + 1
    return chunks
