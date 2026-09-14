param(
    [string]$Executable = ".\dist\ObsidianRAG\ObsidianRAG.exe",
    [int]$Port = 8876
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ExecutableCandidate = if ([System.IO.Path]::IsPathRooted($Executable)) {
    $Executable
} else {
    Join-Path $ProjectRoot $Executable
}
$ExecutablePath = (Resolve-Path -LiteralPath $ExecutableCandidate).Path
$TestRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("ObsidianRAG-bundle-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $TestRoot | Out-Null
$ResolvedTestRoot = (Resolve-Path -LiteralPath $TestRoot).Path
$ExpectedTempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
if (-not $ResolvedTestRoot.StartsWith($ExpectedTempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use a test path outside TEMP: $ResolvedTestRoot"
}

$Process = $null
try {
    $Process = Start-Process -FilePath $ExecutablePath -ArgumentList @(
        '--no-browser', '--port', $Port, '--data-root', "`"$ResolvedTestRoot`""
    ) -WindowStyle Hidden -PassThru
    $Health = $null
    for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
        if ($Process.HasExited) { throw "Product process exited early: $($Process.ExitCode)" }
        try {
            $Health = Invoke-RestMethod "http://127.0.0.1:$Port/api/v1/health" -TimeoutSec 2
            break
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $Health -or $Health.status -ne 'ready') { throw 'Product health check did not pass within 30 seconds.' }
    $Page = Invoke-WebRequest "http://127.0.0.1:$Port/" -UseBasicParsing
    if ($Page.Content -notmatch '<div id="root"></div>') { throw 'The bundle did not serve the web assets.' }
    $Conversation = Invoke-RestMethod "http://127.0.0.1:$Port/api/v1/conversations" `
        -Method Post -ContentType 'application/json; charset=utf-8' -Body '{"title":"Bundle check"}'
    $Stream = Invoke-WebRequest `
        "http://127.0.0.1:$Port/api/v1/conversations/$($Conversation.id)/messages/stream" `
        -Method Post -ContentType 'application/json; charset=utf-8' `
        -Body '{"question":"\u5317\u4eac\u660e\u5929\u5929\u6c14\u600e\u4e48\u6837\uff1f"}' -UseBasicParsing
    $StreamText = if ($Stream.Content -is [byte[]]) {
        [System.Text.Encoding]::UTF8.GetString($Stream.Content)
    } else { [string]$Stream.Content }
    if ($StreamText -notmatch '"type"\s*:\s*"final"' -or
        $StreamText -notmatch '"rejected"\s*:\s*true') {
        throw 'The packaged chat endpoint did not return the expected grounded rejection.'
    }
    Invoke-RestMethod "http://127.0.0.1:$Port/api/v1/system/shutdown" -Method Post | Out-Null
    if (-not $Process.WaitForExit(10000)) { throw 'Product did not exit within 10 seconds.' }
    Write-Host "Bundle smoke test passed: schema=$($Health.database_schema), port=$Port"
} finally {
    if ($Process -and -not $Process.HasExited) { Stop-Process -Id $Process.Id -Force }
    Remove-Item -LiteralPath $ResolvedTestRoot -Recurse -Force
}
