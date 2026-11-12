"""
Определение говорящих (диаризация) через sherpa-onnx.

pyannote-segmentation-3.0 находит отрезки речи и смены голоса, а модель голоса
превращает каждый отрезок в «отпечаток», по которым отрезки группируются
в говорящих. Всё работает на процессоре, PyTorch не нужен.

Модели голоса две, потому что ни одна не справилась со всем сразу (проверка
на монологе 21 мин, диалоге двух голосов и записи с 4 говорящими):
- число людей известно -> NeMo TitaNet-small: размечает все три записи без ошибок;
- число неизвестно -> WeSpeaker ResNet34 с осторожным порогом: монолог не
  дробит и диалог находит верно. TitaNet в этом режиме резал монолог пополам.
  Цена осторожности: похожие голоса могут слиться в одного говорящего.
"""

import bisect
import os
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = f"{RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SEGMENTATION_MEMBER = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
# Да, в названии релиза опечатка («recongition») — так у авторов.
EMBEDDINGS_URL = f"{RELEASES}/speaker-recongition-models"
KNOWN_COUNT_EMBEDDING = "nemo_en_titanet_small.onnx"  # 40 МБ
AUTO_EMBEDDING = "wespeaker_en_voxceleb_resnet34_LM.onnx"  # 26 МБ

# Порог для автоопределения числа говорящих (для WeSpeaker): больше — меньше
# говорящих. При 0.6 монолог распадался на двоих, при 0.7 — нет.
AUTO_THRESHOLD = 0.7

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Turn:
    """Отрезок, где говорит один человек. speaker — номер с 1 по порядку появления."""

    start: float
    end: float
    speaker: int


def _download(url: str, target: Path, title: str) -> None:
    part = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as file:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last_percent = -1
        while chunk := response.read(1 << 20):
            file.write(chunk)
            done += len(chunk)
            percent = done * 100 // total if total else 0
            if total and percent != last_percent:
                last_percent = percent
                print(f"\r  {title}: {percent}%", end="", flush=True)
    print()
    part.replace(target)


class Diarizer:
    def __init__(self, models_dir: Path):
        self.directory = models_dir / "diarization"
        self.threads = min(8, os.cpu_count() or 1)

    def _segmentation(self) -> Path:
        path = self.directory / "pyannote-segmentation-3.0.onnx"
        if path.exists():
            return path
        self.directory.mkdir(parents=True, exist_ok=True)
        print("Скачиваю модель для поиска смены голоса (7 МБ)...")
        archive = self.directory / "segmentation.tar.bz2"
        _download(SEGMENTATION_URL, archive, "загрузка")
        # Из архива берём только саму модель, остальное не нужно.
        part = path.with_name(path.name + ".part")
        with tarfile.open(archive, "r:bz2") as tar:
            source = tar.extractfile(SEGMENTATION_MEMBER)
            if source is None:
                raise RuntimeError("В архиве нет модели сегментации")
            with source, part.open("wb") as target:
                shutil.copyfileobj(source, target)
        part.replace(path)
        archive.unlink()
        return path

    def _embedding(self, name: str) -> Path:
        path = self.directory / name
        if not path.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
            print("Скачиваю модель для различения голосов (30-40 МБ)...")
            _download(f"{EMBEDDINGS_URL}/{name}", path, "загрузка")
        return path

    def _create(self, num_speakers: int | None) -> sherpa_onnx.OfflineSpeakerDiarization:
        if num_speakers:
            embedding = self._embedding(KNOWN_COUNT_EMBEDDING)
            clustering = sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers)
        else:
            embedding = self._embedding(AUTO_EMBEDDING)
            clustering = sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=AUTO_THRESHOLD)
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(self._segmentation())
                ),
                num_threads=self.threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(embedding), num_threads=self.threads
            ),
            clustering=clustering,
        )
        if not config.validate():
            raise RuntimeError("Неверные настройки определения говорящих")
        return sherpa_onnx.OfflineSpeakerDiarization(config)

    def run(self, audio: np.ndarray, num_speakers: int | None) -> list[Turn]:
        """audio — моно 16 кГц float32. num_speakers=None — определить число самому."""
        diarization = self._create(num_speakers)

        last_percent = -1

        def progress(done: int, total: int) -> int:
            nonlocal last_percent
            percent = done * 100 // max(total, 1)
            if percent != last_percent:
                last_percent = percent
                print(f"\r  Определяю говорящих: {percent}%", end="", flush=True)
            return 0

        result = diarization.process(audio, callback=progress).sort_by_start_time()
        print()

        turns = [Turn(segment.start, segment.end, segment.speaker) for segment in result]
        if num_speakers is None:
            turns = _merge_minor_speakers(turns)
        return _renumber(turns)


# В автоматическом режиме голос одного человека иногда распадается на несколько
# «говорящих» с парой коротких реплик. Говорящих, на которых приходится меньше
# этой доли речи, присоединяем к соседу по времени.
MINOR_SPEAKER_SHARE = 0.05


def _merge_minor_speakers(turns: list[Turn]) -> list[Turn]:
    totals: dict[int, float] = {}
    for turn in turns:
        totals[turn.speaker] = totals.get(turn.speaker, 0.0) + turn.end - turn.start
    speech = sum(totals.values())
    minor = {speaker for speaker, total in totals.items() if total < speech * MINOR_SPEAKER_SHARE}
    major = [turn for turn in turns if turn.speaker not in minor]
    if not minor or not major:
        return turns

    merged = []
    for turn in turns:
        if turn.speaker in minor:
            middle = (turn.start + turn.end) / 2
            nearest = min(major, key=lambda other: min(abs(middle - other.start), abs(middle - other.end)))
            turn = Turn(turn.start, turn.end, nearest.speaker)
        merged.append(turn)
    return merged


def _renumber(turns: list[Turn]) -> list[Turn]:
    """Нумерует говорящих по порядку появления: первый заговоривший — «Спикер 1»."""
    numbers: dict[int, int] = {}
    return [
        Turn(turn.start, turn.end, numbers.setdefault(turn.speaker, len(numbers) + 1))
        for turn in turns
    ]


class SpeakerLookup:
    """Быстро находит говорящего по времени: отрезки отсортированы, ищем двоичным поиском."""

    # Сколько отрезков назад проверять: длинная реплика могла начаться раньше
    # нескольких коротких, которые её перекрывают.
    LOOKBACK = 20

    def __init__(self, turns: list[Turn]):
        self.turns = turns
        self.starts = [turn.start for turn in turns]

    def speaker_at(self, start: float, end: float) -> int | None:
        """Говорящий, чей отрезок сильнее всего перекрывает [start, end], иначе ближайший по времени."""
        if not self.turns:
            return None
        index = bisect.bisect_right(self.starts, end)
        nearby = self.turns[max(0, index - self.LOOKBACK): index + 1]
        return _best_speaker(nearby or self.turns, start, end)


def _best_speaker(turns: list[Turn], start: float, end: float) -> int:
    best, best_overlap = None, 0.0
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best, best_overlap = turn.speaker, overlap
    if best is not None:
        return best
    middle = (start + end) / 2
    nearest = min(turns, key=lambda turn: min(abs(middle - turn.start), abs(middle - turn.end)))
    return nearest.speaker
