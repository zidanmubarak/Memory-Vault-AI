param(
    [string]$VenvPath = ".venv-smoke",
    [switch]$UseExistingVenv,
    [switch]$AllowPython313,
    [switch]$SkipDocker,
    [int]$ApiPort = 8000,
    [int]$McpPort = 8001
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Keep smoke output focused on actionable failures.
$env:ANONYMIZED_TELEMETRY = "False"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")
$logDir = Join-Path $repoRoot ".smoke-logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$results = New-Object System.Collections.Generic.List[object]

function Add-StepResult {
    param(
        [string]$Name,
        [string]$Status,
        [double]$DurationSec,
        [string]$Details
    )

    [void]$results.Add(
        [pscustomobject]@{
            Step = $Name
            Status = $Status
            DurationSec = [math]::Round($DurationSec, 2)
            Details = $Details
        }
    )
}

function Get-BootstrapPython {
    $candidates = @()
    $candidates += @{ FilePath = "py"; Arguments = @("-3.13") }

    $candidates += @(
        @{ FilePath = "py"; Arguments = @("-3.12") },
        @{ FilePath = "py"; Arguments = @("-3.11") },
        @{ FilePath = "python"; Arguments = @() }
    )

    $detectedVersions = New-Object System.Collections.Generic.List[string]

    foreach ($candidate in $candidates) {
        if ($candidate.FilePath -eq "py" -and $null -eq (Get-Command py -ErrorAction SilentlyContinue)) {
            continue
        }
        if ($candidate.FilePath -eq "python" -and $null -eq (Get-Command python -ErrorAction SilentlyContinue)) {
            continue
        }

        try {
            $versionOutput = Invoke-External -FilePath $candidate.FilePath -Arguments @($candidate.Arguments + @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")) -CaptureOutput
            $versionText = ($versionOutput -join "").Trim()
            [void]$detectedVersions.Add("$($candidate.FilePath) $versionText")
            $version = [version]("$versionText.0")
            $isSupported = $version.Major -eq 3 -and (
                ($version.Minor -in @(11, 12)) -or ($AllowPython313 -and $version.Minor -eq 13)
            )

            if ($isSupported) {
                return [pscustomobject]@{
                    FilePath = $candidate.FilePath
                    Arguments = $candidate.Arguments
                    Version = $versionText
                }
            }
        }
        catch {
            continue
        }
    }

    $detectedSummary = if ($detectedVersions.Count -gt 0) {
        $detectedVersions -join ", "
    }
    else {
        "none"
    }

    if ($AllowPython313) {
        throw "No supported Python interpreter found. Detected interpreters: $detectedSummary. Fresh smoke install requires Python 3.11, 3.12, or 3.13 when -AllowPython313 is set."
    }

    throw "No supported Python interpreter found. Detected interpreters: $detectedSummary. Fresh smoke install requires Python 3.11 or 3.12. Install Python 3.12 or rerun with -AllowPython313 (may require Microsoft C++ Build Tools for chromadb)."
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    $started = Get-Date
    try {
        $null = & $Action
        $duration = ((Get-Date) - $started).TotalSeconds
        Add-StepResult -Name $Name -Status "PASS" -DurationSec $duration -Details ""
        return $true
    }
    catch {
        $duration = ((Get-Date) - $started).TotalSeconds
        Add-StepResult -Name $Name -Status "FAIL" -DurationSec $duration -Details $_.Exception.Message
        return $false
    }
}

function Add-SkippedStep {
    param(
        [string]$Name,
        [string]$Reason
    )

    Add-StepResult -Name $Name -Status "SKIP" -DurationSec 0 -Details $Reason
}

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$WorkingDirectory = $repoRoot,
        [switch]$CaptureOutput
    )

    Push-Location $WorkingDirectory
    try {
        $normalizeOutput = {
            param([object[]]$RawOutput)

            $lines = New-Object System.Collections.Generic.List[string]
            foreach ($item in $RawOutput) {
                if ($item -is [System.Management.Automation.ErrorRecord]) {
                    [void]$lines.Add($item.ToString())
                }
                else {
                    [void]$lines.Add([string]$item)
                }
            }

            return ,$lines.ToArray()
        }

        if ($CaptureOutput) {
            $previousErrorActionPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                $rawOutput = & $FilePath @Arguments 2>&1
            }
            finally {
                $ErrorActionPreference = $previousErrorActionPreference
            }

            $output = & $normalizeOutput @($rawOutput)
            $exitCode = $LASTEXITCODE
            if ($exitCode -ne 0) {
                $outputText = ($output | Out-String)
                throw "Command failed ($exitCode): $FilePath $($Arguments -join ' ')`n$outputText"
            }
            return ,$output
        }

        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $rawOutput = & $FilePath @Arguments 2>&1
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }

        $output = & $normalizeOutput @($rawOutput)
        $exitCode = $LASTEXITCODE
        if ($null -ne $output) {
            $output | ForEach-Object { Write-Host $_ }
        }
        if ($exitCode -ne 0) {
            throw "Command failed ($exitCode): $FilePath $($Arguments -join ' ')"
        }
    }
    finally {
        Pop-Location
    }
}

function Invoke-WebRequestSafe {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [string]$Method = "Get",
        [int]$TimeoutSec = 0
    )

    $requestParams = @{
        Uri = $Uri
        Method = $Method
    }

    if ($TimeoutSec -gt 0) {
        $requestParams.TimeoutSec = $TimeoutSec
    }

    $webRequestCommand = Get-Command Invoke-WebRequest
    if ($webRequestCommand.Parameters.ContainsKey("UseBasicParsing")) {
        $requestParams.UseBasicParsing = $true
    }

    return Invoke-WebRequest @requestParams
}

function Start-BackgroundProcess {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$WorkingDirectory = $repoRoot
    )

    $stdoutPath = Join-Path $logDir "$Name.stdout.log"
    $stderrPath = Join-Path $logDir "$Name.stderr.log"

    $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $WorkingDirectory -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru

    return [pscustomobject]@{
        Process = $process
        StdoutPath = $stdoutPath
        StderrPath = $stderrPath
        Name = $Name
    }
}

function Stop-BackgroundProcess {
    param(
        $Handle
    )

    if ($null -eq $Handle) {
        return
    }

    if ($null -ne $Handle.Process -and -not $Handle.Process.HasExited) {
        Stop-Process -Id $Handle.Process.Id -Force
    }
}

function Wait-HttpEndpoint {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [int]$TimeoutSeconds = 60,
        [int]$PollIntervalMs = 1000
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $null = Invoke-WebRequestSafe -Uri $Uri -Method "Get" -TimeoutSec 5
            return
        }
        catch {
            Start-Sleep -Milliseconds $PollIntervalMs
        }
    }

    throw "Timed out waiting for endpoint: $Uri"
}

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

function Convert-JsonOutput {
    param(
        [Parameter(Mandatory = $true)][string[]]$OutputLines,
        [Parameter(Mandatory = $true)][string]$Context
    )

    $raw = ($OutputLines -join "`n").Trim()
    if ([string]::IsNullOrWhiteSpace($raw)) {
        throw "$Context returned empty output"
    }

    try {
        return $raw | ConvertFrom-Json
    }
    catch {
        $jsonStart = $raw.IndexOf("{")
        if ($jsonStart -lt 0) {
            $jsonStart = $raw.IndexOf("[")
        }
        if ($jsonStart -lt 0) {
            throw "$Context returned non-JSON output: $raw"
        }

        $jsonOnly = $raw.Substring($jsonStart)
        try {
            return $jsonOnly | ConvertFrom-Json
        }
        catch {
            throw "$Context returned unparsable JSON output: $raw"
        }
    }
}

$venvFullPath = Join-Path $repoRoot $VenvPath
$pythonExe = Join-Path $venvFullPath "Scripts/python.exe"
$memoryLayerExe = Join-Path $venvFullPath "Scripts/memory-layer.exe"

$pythonReady = $false

if ($UseExistingVenv) {
    $pythonReady = Invoke-Step "Use existing venv" {
        Assert-True -Condition (Test-Path $pythonExe) -Message "Python executable not found in $pythonExe"
        Assert-True -Condition (Test-Path $memoryLayerExe) -Message "memory-layer executable not found in $memoryLayerExe"

    $dependencyProbePath = Join-Path $repoRoot ".smoke_dependency_probe.py"
    @'
import importlib.util

required = [
    "fastapi",
    "typer",
    "uvicorn",
    "chromadb",
    "tiktoken",
    "qdrant_client",
    "pydantic",
    "ruff",
    "mypy",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing required modules in existing venv: " + ", ".join(missing))
'@ | Set-Content -Path $dependencyProbePath -Encoding UTF8

        try {
            Invoke-External -FilePath $pythonExe -Arguments @($dependencyProbePath)
        }
        finally {
            if (Test-Path $dependencyProbePath) {
                Remove-Item -Force $dependencyProbePath
            }
        }
    }
}
else {
    $pythonReady = Invoke-Step "Fresh install from scratch" {
        $bootstrapPython = Get-BootstrapPython

        if (Test-Path $venvFullPath) {
            Remove-Item -Recurse -Force $venvFullPath
        }

        Invoke-External -FilePath $bootstrapPython.FilePath -Arguments @($bootstrapPython.Arguments + @("-m", "venv", $venvFullPath))
        Assert-True -Condition (Test-Path $pythonExe) -Message "Failed to create virtual environment"

        Invoke-External -FilePath $pythonExe -Arguments @("-m", "pip", "install", "-U", "pip")
        Invoke-External -FilePath $pythonExe -Arguments @("-m", "pip", "install", "-e", ".[all,dev]")

        Assert-True -Condition (Test-Path $memoryLayerExe) -Message "memory-layer executable not found after install"
    }
}

if ($pythonReady) {
    $null = Invoke-Step "Static quality gate (ruff + mypy)" {
        Invoke-External -FilePath $pythonExe -Arguments @("-m", "ruff", "check", ".", "--exclude", $VenvPath)
        Invoke-External -FilePath $pythonExe -Arguments @("-m", "mypy", ".")
    }

    $null = Invoke-Step "SDK smoke test" {
        $sdkScriptPath = Join-Path $repoRoot ".smoke_sdk_check.py"
        @'
import asyncio

from memory_layer import MemoryLayer


async def main() -> None:
    memory = MemoryLayer(user_id="smoke_user_sdk", session_id="sess_smoke_sdk")
    try:
        await memory.save("I build APIs with FastAPI and PostgreSQL.")
        await memory.save("I prefer concise technical explanations.")

        recall = await memory.recall(
            "What stack does the user use?",
            top_k=5,
            token_budget=512,
        )
        if len(recall.memories) < 1:
            raise RuntimeError("SDK recall returned no memories")

        listed = await memory.list(page=1, page_size=20)
        if listed.total < 1:
            raise RuntimeError("SDK list returned no memories")

        deleted = await memory.forget_all(confirm=True)
        if deleted < 1:
            raise RuntimeError("SDK cleanup did not delete any records")
    finally:
        await memory.close()


asyncio.run(main())
print("SDK smoke OK")
'@ | Set-Content -Path $sdkScriptPath -Encoding UTF8

        try {
            Invoke-External -FilePath $pythonExe -Arguments @($sdkScriptPath)
        }
        finally {
            if (Test-Path $sdkScriptPath) {
                Remove-Item -Force $sdkScriptPath
            }
        }
    }

    $null = Invoke-Step "API + UI + OpenAPI smoke test" {
        $apiHandle = $null
        $apiUser = "smoke_user_api"
        $sessionId = "sess_smoke_api"

        try {
            $apiHandle = Start-BackgroundProcess -Name "api-smoke" -FilePath $pythonExe -Arguments @("-m", "uvicorn", "memory_layer.api.main:app", "--host", "127.0.0.1", "--port", "$ApiPort")

            Wait-HttpEndpoint -Uri "http://127.0.0.1:$ApiPort/v1/health" -TimeoutSeconds 60

            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/v1/health" -Method Get
            Assert-True -Condition ($health.status -eq "ok") -Message "API health status was not ok"

            $saveBody = @{
                user_id = $apiUser
                session_id = $sessionId
                text = "I use FastAPI and PostgreSQL for backend services."
            } | ConvertTo-Json -Compress
            $saveResult = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$ApiPort/v1/memory" -ContentType "application/json" -Body $saveBody
            Assert-True -Condition ($saveResult.saved.Count -ge 1) -Message "API save returned no saved items"

            $query = [uri]::EscapeDataString("what database is used")
            $recallUri = "http://127.0.0.1:$ApiPort/v1/memory/recall?user_id=$apiUser&query=$query&top_k=5&token_budget=512"
            $recall = Invoke-RestMethod -Uri $recallUri -Method Get
            Assert-True -Condition ($recall.memories.Count -ge 1) -Message "API recall returned no memories"

            $list = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/v1/memory?user_id=$apiUser&page=1&page_size=20" -Method Get
            Assert-True -Condition ($list.total -ge 1) -Message "API list returned no records"

            $stats = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/v1/session/$sessionId/stats?user_id=$apiUser" -Method Get
            Assert-True -Condition ($stats.session_id -eq $sessionId) -Message "Session stats returned unexpected session"

            Wait-HttpEndpoint -Uri "http://127.0.0.1:$ApiPort/docs" -TimeoutSeconds 30

            $openapi = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/openapi.json" -Method Get
            Assert-True -Condition ($null -ne $openapi.paths) -Message "OpenAPI schema missing paths"

            Wait-HttpEndpoint -Uri "http://127.0.0.1:$ApiPort/ui" -TimeoutSeconds 30

            $deleteBody = @{ user_id = $apiUser; confirm = $true } | ConvertTo-Json -Compress
            $null = Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:$ApiPort/v1/memory" -ContentType "application/json" -Body $deleteBody
        }
        finally {
            Stop-BackgroundProcess -Handle $apiHandle
        }
    }

    $null = Invoke-Step "CLI smoke test" {
        $cliUser = "smoke_user_cli"

        $seedScriptPath = Join-Path $repoRoot ".smoke_cli_seed.py"
        @'
import asyncio

from memory_layer import MemoryLayer


async def main() -> None:
    memory = MemoryLayer(user_id="smoke_user_cli", session_id="sess_smoke_cli")
    try:
        await memory.save("CLI smoke: FastAPI with PostgreSQL.")
    finally:
        await memory.close()


asyncio.run(main())
print("CLI seed OK")
'@ | Set-Content -Path $seedScriptPath -Encoding UTF8

        try {
            Invoke-External -FilePath $pythonExe -Arguments @($seedScriptPath)
        }
        finally {
            if (Test-Path $seedScriptPath) {
                Remove-Item -Force $seedScriptPath
            }
        }

        Invoke-External -FilePath $memoryLayerExe -Arguments @("--help")

        $listOut = Invoke-External -FilePath $memoryLayerExe -Arguments @("list", "--user-id", $cliUser, "--json") -CaptureOutput
        $listObj = Convert-JsonOutput -OutputLines $listOut -Context "CLI list"
        Assert-True -Condition ($listObj.total -ge 1) -Message "CLI list JSON total was < 1"

        $searchOut = Invoke-External -FilePath $memoryLayerExe -Arguments @("search", "what stack is used", "--user-id", $cliUser, "--json") -CaptureOutput
        $searchObj = Convert-JsonOutput -OutputLines $searchOut -Context "CLI search"
        Assert-True -Condition ($searchObj.memories.Count -ge 1) -Message "CLI search JSON memories were empty"

        $statsOut = Invoke-External -FilePath $memoryLayerExe -Arguments @("stats", "--user-id", $cliUser, "--json") -CaptureOutput
        $statsObj = Convert-JsonOutput -OutputLines $statsOut -Context "CLI stats"
        Assert-True -Condition ($statsObj.total_memories -ge 1) -Message "CLI stats total_memories was < 1"

        $compressOut = Invoke-External -FilePath $memoryLayerExe -Arguments @("compress", "--user-id", $cliUser, "--json") -CaptureOutput
        $compressObj = Convert-JsonOutput -OutputLines $compressOut -Context "CLI compress"
        Assert-True -Condition ($compressObj.user_id -eq $cliUser) -Message "CLI compress returned wrong user_id"

        Invoke-External -FilePath $memoryLayerExe -Arguments @("delete", "--user-id", $cliUser, "--all", "--yes")
    }

    $null = Invoke-Step "MCP smoke test" {
        $toolsOut = Invoke-External -FilePath $memoryLayerExe -Arguments @("mcp", "tools", "--json") -CaptureOutput
        $toolsObj = Convert-JsonOutput -OutputLines $toolsOut -Context "MCP tools"
        Assert-True -Condition ($toolsObj.tools.Count -ge 4) -Message "MCP tools command returned too few tools"

        $mcpHandle = $null
        try {
            $mcpHandle = Start-BackgroundProcess -Name "mcp-smoke" -FilePath $memoryLayerExe -Arguments @("mcp", "start", "--host", "127.0.0.1", "--port", "$McpPort")

            Wait-HttpEndpoint -Uri "http://127.0.0.1:$McpPort/mcp/v1/health" -TimeoutSeconds 60

            $rpcBody = @{ jsonrpc = "2.0"; id = 1; method = "tools/list" } | ConvertTo-Json -Compress
            $rpcResult = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$McpPort/mcp/v1" -ContentType "application/json" -Body $rpcBody

            Assert-True -Condition ($rpcResult.result.tools.Count -ge 4) -Message "MCP JSON-RPC tools/list returned too few tools"
        }
        finally {
            Stop-BackgroundProcess -Handle $mcpHandle
        }
    }

    if ($SkipDocker) {
        Add-SkippedStep -Name "Docker smoke test" -Reason "Skipped by -SkipDocker"
    }
    else {
        $null = Invoke-Step "Docker smoke test" {
            Assert-True -Condition ($null -ne (Get-Command docker -ErrorAction SilentlyContinue)) -Message "docker command not found"

            try {
                Invoke-External -FilePath "docker" -Arguments @("compose", "up", "--build", "-d")
                Wait-HttpEndpoint -Uri "http://127.0.0.1:$ApiPort/v1/health" -TimeoutSeconds 180

                $health = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/v1/health" -Method Get
                Assert-True -Condition ($health.status -eq "ok") -Message "Docker API health was not ok"

                Wait-HttpEndpoint -Uri "http://127.0.0.1:$ApiPort/docs" -TimeoutSeconds 30

                Wait-HttpEndpoint -Uri "http://127.0.0.1:$ApiPort/ui" -TimeoutSeconds 30
            }
            finally {
                try {
                    Invoke-External -FilePath "docker" -Arguments @("compose", "down")
                }
                catch {
                    Write-Warning "docker compose down failed: $($_.Exception.Message)"
                }
            }
        }
    }

    $null = Invoke-Step "Benchmark sanity" {
        $benchmarkOut = Invoke-External -FilePath $pythonExe -Arguments @(
                "scripts/benchmark/run_benchmark_suite.py",
                "--save-count", "50",
                "--recall-count", "20",
                "--warmup-saves", "10",
                "--warmup-recalls", "5",
                "--format", "json"
            ) -CaptureOutput

        $benchmark = (($benchmarkOut -join "`n") | ConvertFrom-Json)
        $operationNames = @($benchmark.operations | ForEach-Object { $_.operation })
        Assert-True -Condition ($operationNames -contains "save") -Message "Benchmark missing save operation"
        Assert-True -Condition ($operationNames -contains "recall") -Message "Benchmark missing recall operation"
    }
}
else {
    Add-SkippedStep -Name "Static quality gate (ruff + mypy)" -Reason "Python environment unavailable"
    Add-SkippedStep -Name "SDK smoke test" -Reason "Python environment unavailable"
    Add-SkippedStep -Name "API + UI + OpenAPI smoke test" -Reason "Python environment unavailable"
    Add-SkippedStep -Name "CLI smoke test" -Reason "Python environment unavailable"
    Add-SkippedStep -Name "MCP smoke test" -Reason "Python environment unavailable"
    if ($SkipDocker) {
        Add-SkippedStep -Name "Docker smoke test" -Reason "Skipped by -SkipDocker"
    }
    else {
        Add-SkippedStep -Name "Docker smoke test" -Reason "Python environment unavailable"
    }
    Add-SkippedStep -Name "Benchmark sanity" -Reason "Python environment unavailable"
}

Write-Host ""
Write-Host "Smoke Checklist Summary" -ForegroundColor Cyan
$results | Format-Table -AutoSize

$failedCount = @($results | Where-Object { $_.Status -eq "FAIL" }).Count
$passCount = @($results | Where-Object { $_.Status -eq "PASS" }).Count
$skipCount = @($results | Where-Object { $_.Status -eq "SKIP" }).Count

if ($failedCount -gt 0) {
    Write-Host ""
    Write-Host "Failure details:" -ForegroundColor Yellow
    foreach ($failedStep in ($results | Where-Object { $_.Status -eq "FAIL" })) {
        Write-Host "- $($failedStep.Step): $($failedStep.Details)"
    }
}

Write-Host ""
Write-Host "Pass: $passCount  Fail: $failedCount  Skip: $skipCount"
Write-Host "Logs: $logDir"

if ($failedCount -gt 0) {
    exit 1
}

exit 0
