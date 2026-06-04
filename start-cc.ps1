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

function Test-SamePath {
    param(
        [string]$Left,
        [string]$Right
    )

    if (-not $Left -or -not $Right) {
        return $false
    }

    try {
        $leftPath = [System.IO.Path]::GetFullPath($Left)
        $rightPath = [System.IO.Path]::GetFullPath($Right)
        return [string]::Equals($leftPath, $rightPath, [System.StringComparison]::OrdinalIgnoreCase)
    } catch {
        return [string]::Equals($Left, $Right, [System.StringComparison]::OrdinalIgnoreCase)
    }
}

function Stop-ProcessAndWait {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$Reason
    )

    if (-not $Process -or $Process.HasExited) {
        return
    }

    Write-Host "Stopping $Reason PID $($Process.Id) ($($Process.ProcessName))..."
    Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    Wait-Process -Id $Process.Id -Timeout 5 -ErrorAction SilentlyContinue
}

function Get-StaleAdapterProcessIds {
    param(
        [int]$Port,
        [string]$ConfigPath,
        [string[]]$BinaryPaths
    )

    $processIds = @()
    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($connections) {
        $processIds += $connections |
            Select-Object -ExpandProperty OwningProcess -Unique |
            Where-Object { $_ -gt 0 }
    }

    $adapterProcesses = Get-CimInstance Win32_Process -Filter "Name = 'claude-adapter.exe'" -ErrorAction SilentlyContinue
    foreach ($adapterProcess in $adapterProcesses) {
        $isCurrentAdapter = $false

        if ($adapterProcess.CommandLine -and $adapterProcess.CommandLine.IndexOf($ConfigPath, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
            $isCurrentAdapter = $true
        }

        if (-not $isCurrentAdapter -and $adapterProcess.ExecutablePath) {
            foreach ($binaryPath in $BinaryPaths) {
                if (Test-SamePath -Left $adapterProcess.ExecutablePath -Right $binaryPath) {
                    $isCurrentAdapter = $true
                    break
                }
            }
        }

        if ($isCurrentAdapter) {
            $processIds += $adapterProcess.ProcessId
        }
    }

    return $processIds | Select-Object -Unique
}

function Stop-StaleAdapterResources {
    param(
        [int]$Port,
        [string]$ConfigPath,
        [string[]]$BinaryPaths
    )

    $processIds = Get-StaleAdapterProcessIds -Port $Port -ConfigPath $ConfigPath -BinaryPaths $BinaryPaths

    foreach ($processId in $processIds) {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if (-not $process) {
            continue
        }

        if ($process.ProcessName -eq "claude-adapter") {
            Stop-ProcessAndWait -Process $process -Reason "stale CC-Adapter resource"
        }
    }

    if (-not (Wait-PortReleased -Port $Port)) {
        Write-Warning "Port $Port was not released after cleaning stale CC-Adapter resources."
    }
}

function Stop-Adapter {
    if ($script:CleanupDone) {
        return
    }
    $script:CleanupDone = $true

    if ($script:AdapterProcess -and -not $script:AdapterProcess.HasExited) {
        Stop-ProcessAndWait -Process $script:AdapterProcess -Reason "CC-Adapter"
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

Stop-StaleAdapterResources -Port $PortValue -ConfigPath $ConfigPath -BinaryPaths $BinaryCandidates

$existing = Get-ListeningProcess -Port $PortValue
if ($existing) {
    Write-Error "Port $PortValue is already in use by PID $($existing.Id) ($($existing.ProcessName)). Stop it first or change config.toml."
    exit 1
}

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
        Write-Host "Stop requested."
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
