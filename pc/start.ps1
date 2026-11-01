$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try { $Host.UI.RawUI.WindowTitle = 'Боксы' } catch {}
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch {}
$Root = Join-Path $env:LOCALAPPDATA 'boxy'
$UvDir = Join-Path $Root 'uv'
$Uv = Join-Path $UvDir 'uv.exe'
$Venv = Join-Path $Root 'venv'
$Py = Join-Path $Venv 'Scripts\python.exe'
$Ready = Join-Path $Root 'ready-1.txt'
$Fw = Join-Path $Root 'firewall-1.txt'
$env:UV_CACHE_DIR = Join-Path $Root 'cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root 'python'
$env:UV_LINK_MODE = 'copy'
$env:BOXY_DATA = $Root
New-Item -ItemType Directory -Force -Path $Root | Out-Null

function Say([string]$Text, [string]$Color = 'Cyan') {
    Write-Host ''
    Write-Host $Text -ForegroundColor $Color
}

function Run([string]$Exe, [string[]]$Arguments, [string]$Fail) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw $Fail }
}

function GpuOk {
    $ErrorActionPreference = 'Continue'
    & $Py -c "import sys,torch;x=torch.ones(1,3,16,16,device='cuda',dtype=torch.float16);w=torch.ones(4,3,3,3,device='cuda',dtype=torch.float16);sys.exit(0 if torch.nn.functional.conv2d(x,w).sum().item()>0 else 1)" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

try {
    if (-not (Test-Path $Uv)) {
        Say 'Скачиваю установщик (uv)...'
        $zip = Join-Path $Root 'uv.zip'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath $UvDir -Force
        Remove-Item $zip -Force
        if (-not (Test-Path $Uv)) {
            $found = Get-ChildItem -Path $UvDir -Recurse -Filter 'uv.exe' | Select-Object -First 1
            if ($found) { Copy-Item $found.FullName $Uv -Force }
        }
        if (-not (Test-Path $Uv)) { throw 'uv.exe не найден после распаковки' }
    }
    if (-not (Test-Path $Ready)) {
        Say 'Первый запуск: ставлю Python, PyTorch для видеокарты и YOLO (около 4 ГБ, 5-15 минут).'
        if (-not (Test-Path $Py)) {
            if (Test-Path $Venv) { Remove-Item $Venv -Recurse -Force }
            Run $Uv @('venv', $Venv, '--python', '3.13') 'не удалось создать окружение Python'
        }
        Run $Uv @('pip', 'install', '--python', $Py, 'torch', 'torchvision', 'ultralytics', 'aiohttp', 'cryptography', '--torch-backend=auto') 'не удалось установить пакеты'
        if (-not (GpuOk)) {
            Say 'Видеокарта не ответила, ставлю PyTorch для CUDA 12.8...'
            Run $Uv @('pip', 'install', '--python', $Py, '--reinstall-package', 'torch', '--reinstall-package', 'torchvision', 'torch', 'torchvision', '--torch-backend=cu128') 'не удалось установить PyTorch'
            if (-not (GpuOk)) { Say 'ВНИМАНИЕ: видеокарта так и не ответила. Обновите драйвер NVIDIA и запустите start.bat снова. Пока сервер будет работать на процессоре.' 'Yellow' }
        }
        Set-Content -Path $Ready -Value (Get-Date -Format s) -Encoding ASCII
    }
    if (-not (Test-Path $Fw)) {
        $base = ([string](& $Py -c "import sys;print(sys._base_executable)")).Trim()
        $progs = @($base, $Py) | Where-Object { $_ } | Select-Object -Unique
        $lines = @("Get-NetFirewallRule -DisplayName 'Boxy' -ErrorAction SilentlyContinue | Remove-NetFirewallRule")
        foreach ($p in $progs) {
            $lines += "Get-NetFirewallApplicationFilter -Program '$p' -ErrorAction SilentlyContinue | Get-NetFirewallRule | Where-Object { `$_.Action -eq 'Block' } | Remove-NetFirewallRule"
            $lines += "New-NetFirewallRule -DisplayName 'Boxy' -Direction Inbound -Action Allow -Program '$p' -Profile Any | Out-Null"
        }
        $enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($lines -join '; '))
        Say 'Сейчас Windows спросит права администратора: это разрешение в брандмауэре, чтобы телефон мог подключиться к ПК. Нажмите «Да».'
        try {
            Start-Process -FilePath 'powershell.exe' -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList "-NoProfile -EncodedCommand $enc"
            Set-Content -Path $Fw -Value 'ok' -Encoding ASCII
        } catch {
            Say 'Права не выданы. Если появится окно брандмауэра для Python, отметьте обе галочки и нажмите «Разрешить доступ». В следующий запуск спрошу снова.' 'Yellow'
        }
    }
    Say 'Запускаю сервер. Не закрывайте это окно, пока пользуетесь камерой.'
    & $Py -u (Join-Path $PSScriptRoot 'server.py')
} catch {
    Say ('ОШИБКА: ' + $_.Exception.Message) 'Red'
    exit 1
}
