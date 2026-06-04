param(
    [int]$StopAfterSeconds = 0
)

$ErrorActionPreference = "Stop"
$script:StopRequested = $false
$script:CleanupDone = $false
$script:AdapterProcess = $null

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $ProjectRoot "config.toml"
$LogDir = Join-Path $ProjectRoot "logs"
$OutLog = Join-Path $LogDir "cc-adapter.out.log"
$ErrLog = Join-Path $LogDir "cc-adapter.err.log"

function Get-ConfigValue {
    param(
        [string]$Text,
        [string]$Name,
        [string]$DefaultValue
    )

    if ($Text -match "(?m)^\s*$Name\s*=\s*`"([^`"]+)`"") {
        return $Matches[1]
    }
    if ($Text -match "(?m)^\s*$Name\s*=\s*(\d+)") {
        return $Matches[1]
    }
    return $DefaultValue
}

function Get-ListeningProcess {
    param([int]$Port)

    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $conn) {
        return $null
    }
    return Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
}

function Wait-PortReleased {
    param(
        [int]$Port,
        [int]$TimeoutMs = 5000
    )

    $deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMs)
    while ([DateTime]::UtcNow -lt $deadline) {
        $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $conn) {
            return $true
        }
        Start-Sleep -Milliseconds 100
    }
    return $false
}

function Stop-Adapter {
    if ($script:CleanupDone) {
        return
    }
    $script:CleanupDone = $true

    if ($script:AdapterProcess -and -not $script:AdapterProcess.HasExited) {
        Write-Host "Stopping CC-Adapter..."
        Stop-Process -Id $script:AdapterProcess.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $script:AdapterProcess.Id -Timeout 5 -ErrorAction SilentlyContinue
    }
}

if (-not (Test-Path $ConfigPath)) {
    Write-Error "Missing config file: $ConfigPath"
    exit 1
}

$BinaryCandidates = @(
    (Join-Path $ProjectRoot "target\release\claude-adapter.exe"),
    (Join-Path $ProjectRoot "target\x86_64-pc-windows-gnu\release\claude-adapter.exe"),
    (Join-Path $ProjectRoot "release\claude-adapter.exe"),
    (Join-Path $ProjectRoot "claude-adapter.exe")
)

$AdapterExe = $BinaryCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $AdapterExe) {
    Write-Error "Missing claude-adapter executable. Build with 'cargo build --release' or put claude-adapter.exe under release\."
    exit 1
}

$ConfigText = Get-Content $ConfigPath -Raw
$HostValue = Get-ConfigValue -Text $ConfigText -Name "host" -DefaultValue "127.0.0.1"
$PortValue = [int](Get-ConfigValue -Text $ConfigText -Name "port" -DefaultValue "8080")

$BaseUrl = "http://${HostValue}:${PortValue}"
if ($HostValue -eq "0.0.0.0") {
    $BaseUrl = "http://127.0.0.1:${PortValue}"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$existing = Get-ListeningProcess -Port $PortValue
if ($existing) {
    Write-Error "Port $PortValue is already in use by PID $($existing.Id) ($($existing.ProcessName)). Stop it first or change config.toml."
    exit 1
}

[Console]::add_CancelKeyPress({
    param($sender, $eventArgs)
    $eventArgs.Cancel = $true
    $script:StopRequested = $true
})

Write-Host "Starting CC-Adapter at $BaseUrl"
Write-Host "Logs:"
Write-Host "  stdout: $OutLog"
Write-Host "  stderr: $ErrLog"

$script:AdapterProcess = Start-Process -FilePath $AdapterExe `
    -ArgumentList @("serve", "--config", $ConfigPath) `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -PassThru

$ready = $false
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 250
    if ($script:AdapterProcess.HasExited) {
        break
    }
    $listener = Get-ListeningProcess -Port $PortValue
    if ($listener -and $listener.Id -eq $script:AdapterProcess.Id) {
        $ready = $true
        break
    }
}

if (-not $ready) {
    Write-Host "CC-Adapter failed to start. stderr:"
    if (Test-Path $ErrLog) {
        Get-Content $ErrLog -Tail 40
    }
    Stop-Adapter
    exit 1
}

try {
    Write-Host "CC-Adapter is running at $BaseUrl"
    Write-Host "Process ID: $($script:AdapterProcess.Id)"
    Write-Host ""
    Write-Host "Copy and run the following commands in a NEW PowerShell terminal, then type: claude"
    Write-Host ""
    Write-Host "Remove-Item Env:ANTHROPIC_MODEL -ErrorAction SilentlyContinue"
    Write-Host "Remove-Item Env:ANTHROPIC_BASE_URL -ErrorAction SilentlyContinue"
    Write-Host "Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue"
    Write-Host "Remove-Item Env:ANTHROPIC_CUSTOM_HEADERS -ErrorAction SilentlyContinue"
    Write-Host "`$env:ANTHROPIC_BASE_URL = `"$BaseUrl`""
    Write-Host "`$env:ANTHROPIC_API_KEY = `"dummy`""
    Write-Host ""
    Write-Host "Press Ctrl+C to stop CC-Adapter and release port $PortValue."

    $startedAt = [DateTime]::UtcNow
    while (-not $script:StopRequested -and -not $script:AdapterProcess.HasExited) {
        if ($StopAfterSeconds -gt 0 -and [DateTime]::UtcNow -ge $startedAt.AddSeconds($StopAfterSeconds)) {
            Write-Host "StopAfterSeconds reached."
            $script:StopRequested = $true
            break
        }
        Start-Sleep -Seconds 1
    }

    if ($script:StopRequested) {
        Write-Host "Ctrl+C received."
    } elseif ($script:AdapterProcess.HasExited) {
        Write-Host "CC-Adapter exited with code $($script:AdapterProcess.ExitCode)."
    }

    exit 0
} finally {
    Stop-Adapter

    if (Wait-PortReleased -Port $PortValue) {
        Write-Host "Port $PortValue released."
    } else {
        Write-Warning "Port $PortValue is still in use. Check the process manually."
    }
}
