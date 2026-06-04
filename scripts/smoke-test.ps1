param(
    [int]$MockPort = 19081,
    [int]$AdapterPort = 19082
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Exe = Join-Path $ProjectRoot "target\x86_64-pc-windows-gnu\release\claude-adapter.exe"
$LogDir = Join-Path $ProjectRoot "logs"
$MockLog = Join-Path $LogDir "mock-upstream-bodies.jsonl"
$TempConfig = Join-Path $LogDir "mock-config.toml"
$AdapterOut = Join-Path $LogDir "mock-adapter.out.log"
$AdapterErr = Join-Path $LogDir "mock-adapter.err.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Remove-Item $MockLog, $TempConfig, $AdapterOut, $AdapterErr -ErrorAction SilentlyContinue

@"
[server]
host = "127.0.0.1"
port = $AdapterPort
log_level = "debug"
log_file_enabled = false
claude_stream_idle_timeout_ms = 0

[providers.mock]
type = "openai"
api_key = "mock-key"
base_url = "http://127.0.0.1:$MockPort/v1"
supports_streaming = true

[models]
default_provider = "mock"
default_model = "mock-model"

[models.routing]
"claude" = { provider = "mock", model = "mock-model" }
"claude-sonnet-4-6" = { provider = "mock", model = "mock-model" }
"@ | Set-Content -Path $TempConfig -Encoding UTF8

$mockJob = Start-Job -ArgumentList $MockPort, $MockLog -ScriptBlock {
    param($Port, $Log)

    function Write-Utf8($Stream, [string]$Text) {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        $Stream.Write($bytes, 0, $bytes.Length)
        $Stream.Flush()
    }

    function Sse([object]$Object) {
        "data: $($Object | ConvertTo-Json -Depth 50 -Compress)`n`n"
    }

    $listener = [System.Net.HttpListener]::new()
    $listener.Prefixes.Add("http://127.0.0.1:$Port/")
    $listener.Start()

    try {
        for ($i = 0; $i -lt 8; $i++) {
            $ctx = $listener.GetContext()
            $req = $ctx.Request
            $reader = [System.IO.StreamReader]::new($req.InputStream, $req.ContentEncoding)
            $body = $reader.ReadToEnd()
            $reader.Close()
            Add-Content -Path $Log -Value $body

            $json = $body | ConvertFrom-Json
            $isStream = [bool]$json.stream
            $hasTools = $null -ne $json.tools -and $json.tools.Count -gt 0
            $wantTool = $hasTools -and (($json.messages | ConvertTo-Json -Depth 50 -Compress) -match 'Read|tool|use')

            $resp = $ctx.Response
            if ($isStream) {
                $resp.ContentType = "text/event-stream; charset=utf-8"

                if ($wantTool) {
                    Write-Utf8 $resp.OutputStream (Sse @{
                        choices = @(@{
                            index = 0
                            delta = @{
                                tool_calls = @(@{
                                    index = 0
                                    id = "call_read"
                                    function = @{
                                        name = "Read"
                                        arguments = '{"file_path":"README.md",'
                                    }
                                })
                            }
                            finish_reason = $null
                        })
                    })
                    Write-Utf8 $resp.OutputStream (Sse @{
                        choices = @(@{
                            index = 0
                            delta = @{
                                tool_calls = @(@{
                                    index = 0
                                    function = @{
                                        arguments = '"pages":"1","limit":10}'
                                    }
                                })
                            }
                            finish_reason = $null
                        })
                    })
                    Write-Utf8 $resp.OutputStream (Sse @{
                        choices = @(@{
                            index = 0
                            delta = @{}
                            finish_reason = "tool_calls"
                        })
                    })
                } else {
                    Write-Utf8 $resp.OutputStream (Sse @{
                        choices = @(@{
                            index = 0
                            delta = @{ content = "mock-stream-" }
                            finish_reason = $null
                        })
                    })
                    Write-Utf8 $resp.OutputStream (Sse @{
                        choices = @(@{
                            index = 0
                            delta = @{ content = "ok" }
                            finish_reason = $null
                        })
                    })
                    Write-Utf8 $resp.OutputStream (Sse @{
                        choices = @(@{
                            index = 0
                            delta = @{}
                            finish_reason = "stop"
                        })
                    })
                }

                Write-Utf8 $resp.OutputStream "data: [DONE]`n`n"
            } else {
                $resp.ContentType = "application/json; charset=utf-8"
                if ($wantTool) {
                    $text = @{
                        id = "chatcmpl-mock"
                        object = "chat.completion"
                        created = 1
                        model = "mock-model"
                        choices = @(@{
                            index = 0
                            message = @{
                                role = "assistant"
                                content = $null
                                tool_calls = @(@{
                                    id = "call_read"
                                    type = "function"
                                    function = @{
                                        name = "Read"
                                        arguments = '{"file_path":"README.md","pages":"1","limit":10}'
                                    }
                                })
                            }
                            finish_reason = "tool_calls"
                        })
                        usage = @{
                            prompt_tokens = 10
                            completion_tokens = 5
                            total_tokens = 15
                        }
                    } | ConvertTo-Json -Depth 50 -Compress
                } else {
                    $text = @{
                        id = "chatcmpl-mock"
                        object = "chat.completion"
                        created = 1
                        model = "mock-model"
                        choices = @(@{
                            index = 0
                            message = @{ role = "assistant"; content = "mock-nonstream-ok" }
                            finish_reason = "stop"
                        })
                        usage = @{
                            prompt_tokens = 10
                            completion_tokens = 5
                            total_tokens = 15
                        }
                    } | ConvertTo-Json -Depth 50 -Compress
                }
                Write-Utf8 $resp.OutputStream $text
            }
            $resp.Close()
        }
    } finally {
        $listener.Stop()
        $listener.Close()
    }
}

$adapter = Start-Process -FilePath $Exe `
    -ArgumentList @("serve", "--config", $TempConfig) `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $AdapterOut `
    -RedirectStandardError $AdapterErr `
    -WindowStyle Hidden `
    -PassThru

try {
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 250
        if (Get-NetTCPConnection -LocalPort $AdapterPort -State Listen -ErrorAction SilentlyContinue) {
            $ready = $true
            break
        }
        if ($adapter.HasExited) { break }
    }
    if (-not $ready) {
        Get-Content $AdapterErr -Tail 80 -ErrorAction SilentlyContinue
        throw "adapter not ready"
    }

    $headers = @{
        "x-api-key" = "dummy"
        "anthropic-version" = "2023-06-01"
        "content-type" = "application/json"
    }

    $nonstreamBody = @{
        model = "claude-sonnet-4-6"
        max_tokens = 64
        stream = $false
        messages = @(@{ role = "user"; content = "hello" })
    } | ConvertTo-Json -Depth 50
    $nonstream = Invoke-WebRequest -Uri "http://127.0.0.1:$AdapterPort/v1/messages" -Method Post -Headers $headers -Body $nonstreamBody -TimeoutSec 60
    $nonstreamJson = $nonstream.Content | ConvertFrom-Json
    if ($nonstreamJson.content[0].text -ne "mock-nonstream-ok") {
        throw "nonstream failed: $($nonstream.Content)"
    }
    "NONSTREAM_OK status=$($nonstream.StatusCode) text=$($nonstreamJson.content[0].text)"

    $streamBody = @{
        model = "claude-sonnet-4-6"
        max_tokens = 64
        stream = $true
        messages = @(@{ role = "user"; content = "hello stream" })
    } | ConvertTo-Json -Depth 50
    $stream = Invoke-WebRequest -Uri "http://127.0.0.1:$AdapterPort/v1/messages" -Method Post -Headers $headers -Body $streamBody -TimeoutSec 60
    if ($stream.Content -notmatch "content_block_delta" -or $stream.Content -notmatch "mock-stream-" -or $stream.Content -notmatch '"ok"') {
        throw "stream failed: $($stream.Content)"
    }
    "STREAM_OK status=$($stream.StatusCode) contentType=$($stream.Headers["Content-Type"])"

    $toolBody = @{
        model = "claude-sonnet-4-6"
        max_tokens = 64
        stream = $false
        messages = @(@{ role = "user"; content = "hello" })
        tools = @(@{ name = "WebSearch"; description = "Search web" })
    } | ConvertTo-Json -Depth 50
    Invoke-WebRequest -Uri "http://127.0.0.1:$AdapterPort/v1/messages" -Method Post -Headers $headers -Body $toolBody -TimeoutSec 60 | Out-Null
    $lastBody = Get-Content $MockLog | Select-Object -Last 1 | ConvertFrom-Json
    if ($lastBody.tools[0].function.parameters.type -ne "object") {
        throw "missing schema normalizer failed"
    }
    "MISSING_SCHEMA_OK normalizedType=$($lastBody.tools[0].function.parameters.type)"

    $readTool = @{
        name = "Read"
        description = "Read file"
        input_schema = @{
            type = "object"
            properties = @{
                file_path = @{ type = "string" }
                pages = @{ type = "string" }
                limit = @{ type = "number" }
            }
        }
    }
    $readBody = @{
        model = "claude-sonnet-4-6"
        max_tokens = 64
        stream = $true
        messages = @(@{ role = "user"; content = "use Read tool" })
        tools = @($readTool)
    } | ConvertTo-Json -Depth 50
    $readResp = Invoke-WebRequest -Uri "http://127.0.0.1:$AdapterPort/v1/messages" -Method Post -Headers $headers -Body $readBody -TimeoutSec 60
    if ($readResp.Content -notmatch "tool_use" -or $readResp.Content -notmatch "README.md") {
        throw "stream tool failed: $($readResp.Content)"
    }
    if ($readResp.Content -match "pages") {
        throw "Read.pages sanitizer failed: $($readResp.Content)"
    }
    "STREAM_TOOL_SANITIZER_OK"

    $upstreamBodies = Get-Content $MockLog | ForEach-Object { $_ | ConvertFrom-Json }
    $streamFlags = @($upstreamBodies | ForEach-Object { $_.stream }) -join ","
    "UPSTREAM_STREAM_FLAGS=$streamFlags"
} finally {
    if ($adapter -and -not $adapter.HasExited) {
        Stop-Process -Id $adapter.Id -Force
    }
    if ($mockJob) {
        Stop-Job $mockJob -ErrorAction SilentlyContinue
        Remove-Job $mockJob -Force -ErrorAction SilentlyContinue
    }
}
