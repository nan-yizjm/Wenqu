param(
    [string]$Installer = ".\dist-installer\ObsidianRAG-Setup-0.2.0-win-x64.exe",
    [int]$Port = 8879
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$InstallerPath = (Resolve-Path (Join-Path $ProjectRoot $Installer)).Path
$TempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$RunId = [guid]::NewGuid().ToString('N').Substring(0, 8)
# Keep the test path close to the real per-user install path length. Some PyTorch
# license files are deeply nested and Windows installers still encounter MAX_PATH.
$InstallRoot = Join-Path $TempBase "ORAG-$RunId"
$DataRoot = Join-Path $TempBase "ORAG-data-$RunId"
$InstallLog = Join-Path $TempBase "ObsidianRAG-install-$RunId.log"

foreach ($Path in @($InstallRoot, $DataRoot)) {
    $FullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $FullPath.StartsWith($TempBase, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to use a path outside TEMP: $FullPath"
    }
}

$Install = Start-Process -FilePath $InstallerPath -ArgumentList @(
    '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/NOICONS',
    "/DIR=$InstallRoot", "/LOG=$InstallLog"
) -WindowStyle Hidden -PassThru -Wait
if ($Install.ExitCode -ne 0) {
    if (Test-Path -LiteralPath $InstallLog) { Get-Content -LiteralPath $InstallLog -Tail 30 }
    throw "Installer failed: $($Install.ExitCode)"
}
$ExpectedDataRetention = $false
try {
    & (Join-Path $ProjectRoot 'scripts\test_product_bundle.ps1') `
        -Executable (Join-Path $InstallRoot 'ObsidianRAG.exe') -Port $Port `
        -DataRoot $DataRoot -ExpectedMinimumConversations 0
    $ExpectedDataRetention = $true
    $Upgrade = Start-Process -FilePath $InstallerPath -ArgumentList @(
        '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/NOICONS',
        "/DIR=$InstallRoot", "/LOG=$InstallLog"
    ) -WindowStyle Hidden -PassThru -Wait
    if ($Upgrade.ExitCode -ne 0) { throw "Upgrade install failed: $($Upgrade.ExitCode)" }
    & (Join-Path $ProjectRoot 'scripts\test_product_bundle.ps1') `
        -Executable (Join-Path $InstallRoot 'ObsidianRAG.exe') -Port ($Port + 1) `
        -DataRoot $DataRoot -ExpectedMinimumConversations 1
    Write-Host "Installer and data-preserving upgrade validation passed: $InstallRoot"
} finally {
    $Uninstaller = Join-Path $InstallRoot 'unins000.exe'
    if (Test-Path -LiteralPath $Uninstaller) {
        $Uninstall = Start-Process -FilePath $Uninstaller -ArgumentList @(
            '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART'
        ) -WindowStyle Hidden -PassThru -Wait
        if ($Uninstall.ExitCode -ne 0) { throw "Uninstaller failed: $($Uninstall.ExitCode)" }
    }
    if ($ExpectedDataRetention -and
        -not (Test-Path -LiteralPath (Join-Path $DataRoot 'workspace.sqlite3'))) {
        throw 'Uninstall unexpectedly removed user data.'
    }
    if (Test-Path -LiteralPath $DataRoot) { Remove-Item -LiteralPath $DataRoot -Recurse -Force }
    if (Test-Path -LiteralPath $InstallLog) { Remove-Item -LiteralPath $InstallLog -Force }
}
