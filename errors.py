"""
Понятные объяснения ошибок.

Программой пользуются люди, далёкие от компьютеров, поэтому вместо английского
текста исключения показываем, что случилось и что с этим делать. Технический
текст остаётся только у незнакомых ошибок — чтобы его можно было переслать
тому, кто помогает с программой.
"""

import errno


class UserError(Exception):
    """Ошибка, текст которой уже написан для человека: показываем как есть."""


# Имена классов исключений, которые означают проблему с сетью. Сравниваем по
# именам, чтобы не импортировать requests, httpx и huggingface_hub: какая из
# библиотек качает модель, зависит от версий.
_NETWORK_ERRORS = {
    "URLError", "HTTPError", "gaierror", "SSLError",
    "ConnectionError", "ConnectError", "ConnectTimeout", "ReadTimeout", "TimeoutError",
    "RemoteDisconnected", "IncompleteRead", "ProtocolError",
    "HfHubHTTPError", "LocalEntryNotFoundError", "OfflineModeIsEnabled",
}

# Windows сообщает о нехватке места своим кодом, а не errno.
_WINERROR_DISK_FULL = 112


def _chain(error: BaseException) -> list[BaseException]:
    """Ошибка вместе с теми, из-за которых она возникла."""
    found: list[BaseException] = []
    while error is not None and error not in found:
        found.append(error)
        error = error.__cause__ or error.__context__
    return found


def _class_names(errors: list[BaseException]) -> set[str]:
    return {cls.__name__ for error in errors for cls in type(error).__mro__}


def is_out_of_memory(error: BaseException) -> bool:
    """Не хватило памяти — обычной или видеопамяти."""
    for item in _chain(error):
        if isinstance(item, MemoryError):
            return True
        text = str(item).lower()
        if "out of memory" in text or "bad alloc" in text:
            return True
    return False


def explain(error: BaseException) -> str:
    """Объясняет ошибку простыми словами."""
    if isinstance(error, UserError):
        return str(error)

    chain = _chain(error)
    names = _class_names(chain)

    for item in chain:
        if isinstance(item, OSError) and (
            item.errno == errno.ENOSPC or getattr(item, "winerror", None) == _WINERROR_DISK_FULL
        ):
            return (
                "На диске закончилось место. Освободите несколько гигабайт "
                "и запустите распознавание ещё раз."
            )

    if is_out_of_memory(error):
        return (
            "Компьютеру не хватило памяти. Закройте другие программы "
            "или выберите модель поменьше (например, small) и попробуйте ещё раз."
        )

    if names & _NETWORK_ERRORS:
        return (
            "Не получилось скачать модель из интернета. Проверьте, что интернет "
            "работает, и запустите ещё раз: скачивание нужно только в первый раз."
        )

    if "PermissionError" in names:
        return (
            "Нет доступа к файлу или папке. Закройте программы, в которых открыта "
            "запись, её текст или субтитры, и проверьте, что в папку с записью "
            "можно сохранять файлы."
        )

    if "FileNotFoundError" in names:
        return (
            "Файл пропал во время работы. Проверьте, что запись лежит на месте "
            "и что флешка или диск с ней не отключены."
        )

    return (
        "Что-то пошло не так. Покажите эту строку тому, кто помогает вам с программой:\n"
        f"  {type(error).__name__}: {error}"
    )
