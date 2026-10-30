param([switch]$Install)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_XET = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
    $Install = $true
}
if ($Install) {
    & $taskPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
    & $taskPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
    if (Test-Path -LiteralPath 'requirements.lock.txt') {
        & $taskPython -m pip install -r requirements.lock.txt
    } else {
        & $taskPython -m pip install -e .
    }
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $taskPython -m pip install --no-deps -e .
    if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
}
& $taskPython -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
