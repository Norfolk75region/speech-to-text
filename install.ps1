# Установка всего нужного для распознавания речи.
# Ничего не ставит в систему: uv, Python и библиотеки кладутся в папку программы
# (tools\ и .venv\). Запускается через install.bat.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # без полосы прогресса загрузка в PowerShell 5.1 в разы быстрее
Set-Location -LiteralPath $PSScriptRoot

$PythonVersion = "3.12"
$ToolsDir = Join-Path $PSScriptRoot "tools"
$UvDir = Join-Path $ToolsDir "uv"
$UvExe = Join-Path $UvDir "uv.exe"
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

# Python, который скачает uv, тоже держим внутри папки программы.
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ToolsDir "python"

function Write-Step($Text) {
    Write-Host ""
    Write-Host "== $Text" -ForegroundColor Cyan
}

function Test-VenvWorks {
    if (-not (Test-Path -LiteralPath $VenvPython)) { return $false }
    try {
        & $VenvPython -c "import sys" 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

try {
    Write-Host "=== Установка программы распознавания речи ===" -ForegroundColor Green
    Write-Host "Нужен интернет. Скачается около 150 МБ, обычно это занимает пару минут."

    # 1. uv — менеджер Python-пакетов, скачиваем его один раз.
    if (-not (Test-Path -LiteralPath $UvExe)) {
        Write-Step "Скачиваю установщик пакетов uv..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
        $Zip = Join-Path $ToolsDir "uv.zip"
        Invoke-WebRequest -UseBasicParsing `
            -Uri "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" `
            -OutFile $Zip
        Expand-Archive -LiteralPath $Zip -DestinationPath $UvDir -Force
        Remove-Item -LiteralPath $Zip
        if (-not (Test-Path -LiteralPath $UvExe)) {
            # На случай, если в архиве uv.exe лежит во вложенной папке.
            $Found = Get-ChildItem -LiteralPath $UvDir -Recurse -Filter "uv.exe" | Select-Object -First 1
            if (-not $Found) { throw "В скачанном архиве нет uv.exe" }
            Move-Item -LiteralPath $Found.FullName -Destination $UvExe
        }
    } else {
        Write-Step "Установщик пакетов uv уже есть."
    }

    # 2. Python и виртуальное окружение. Если папку программы перенесли,
    #    старое окружение перестаёт работать — тогда создаём его заново.
    if (Test-VenvWorks) {
        Write-Step "Python уже установлен."
    } else {
        Write-Step "Скачиваю Python $PythonVersion..."
        & $UvExe venv --clear --python $PythonVersion --python-preference only-managed .venv
        if ($LASTEXITCODE -ne 0) { throw "Не удалось установить Python" }
    }

    # 3. Библиотеки для распознавания.
    Write-Step "Скачиваю библиотеки для распознавания речи..."
    & $UvExe pip install --python $VenvPython -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Не удалось установить библиотеки" }

    # 4. Проверка.
    Write-Step "Проверяю установку..."
    & $VenvPython main.py --list-models
    if ($LASTEXITCODE -ne 0) { throw "Программа не запускается после установки" }

    Write-Host ""
    Write-Host "Установка завершена. Теперь запускайте run.bat." -ForegroundColor Green
    Write-Host "Модель распознавания скачается при первом запуске."
    exit 0
} catch {
    Write-Host ""
    Write-Host "Ошибка установки: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Проверьте интернет и запустите install.bat ещё раз."
    Write-Host "Если не поможет, пришлите снимок этого окна тому, кто дал вам программу."
    exit 1
}
