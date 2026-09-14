param(
    [string]$Executable = ".\dist\ObsidianRAG\ObsidianRAG.exe",
    [string]$DataRoot = "$env:LOCALAPPDATA\ObsidianRAG",
    [int]$Port = 8878
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ExecutablePath = (Resolve-Path (Join-Path $ProjectRoot $Executable)).Path
$ResolvedDataRoot = [System.IO.Path]::GetFullPath($DataRoot)
$Process = Start-Process -FilePath $ExecutablePath -ArgumentList @(
    '--no-browser', '--port', $Port, '--data-root', "`"$ResolvedDataRoot`""
) -WindowStyle Hidden -PassThru
try {
    $Health = $null
    for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
        if ($Process.HasExited) { throw "Product process exited early: $($Process.ExitCode)" }
        try {
            $Health = Invoke-RestMethod "http://127.0.0.1:$Port/api/v1/health" -TimeoutSec 2
            break
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $Health) { throw 'Product health check failed.' }
    Invoke-RestMethod "http://127.0.0.1:$Port/api/v1/setup/retrieval-model" -Method Post | Out-Null
    $Model = $null
    for ($Attempt = 0; $Attempt -lt 180; $Attempt++) {
        Start-Sleep -Seconds 1
        $Model = Invoke-RestMethod "http://127.0.0.1:$Port/api/v1/setup/retrieval-model"
        if ($Model.status -in @('ready', 'failed')) { break }
    }
    if (-not $Model -or $Model.status -ne 'ready') {
        throw "Packaged retrieval model validation failed: $($Model.detail)"
    }
    if ($Model.dimension -ne 384 -or $Model.device -ne 'cpu') {
        throw 'Packaged retrieval model returned unexpected metadata.'
    }
    Write-Host "Packaged E5 validation passed: dimension=$($Model.dimension), device=$($Model.device)"
    Invoke-RestMethod "http://127.0.0.1:$Port/api/v1/system/shutdown" -Method Post | Out-Null
    if (-not $Process.WaitForExit(15000)) { throw 'Product did not exit within 15 seconds.' }
} finally {
    if (-not $Process.HasExited) { Stop-Process -Id $Process.Id -Force }
}
