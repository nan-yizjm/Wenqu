param(
    [string]$Installer = "",
    [string]$LegacyBundle = "",
    [int]$Port = 8879
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$TempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$RunId = [guid]::NewGuid().ToString('N').Substring(0, 8)
# Keep the test path close to the real per-user install path length. Some PyTorch
# license files are deeply nested and Windows installers still encounter MAX_PATH.
$InstallRoot = Join-Path $TempBase "ORAG-$RunId"
$DataRoot = Join-Path $TempBase "ORAG-data-$RunId"
$InstallLog = Join-Path $TempBase "ObsidianRAG-install-$RunId.log"
$TestOutput = Join-Path $TempBase "Wenqu-acceptance-$RunId"
# Dedicated AppId prevents test installs overwriting the user's uninstall registry.
if ([string]::IsNullOrWhiteSpace($Installer)) {
    New-Item -ItemType Directory -Path $TestOutput | Out-Null
    $Compiler = Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe'
    if (-not (Test-Path -LiteralPath $Compiler)) { $Compiler = 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe' }
    & $Compiler '/Qp' "/DMyAppId=WenquAcceptance-$RunId" '/DTestBuild=1' "/DInstallerOutputDir=$TestOutput" (Join-Path $ProjectRoot 'installer\Wenqu.iss')
    if ($LASTEXITCODE -ne 0) { throw 'Test installer compilation failed.' }
    $InstallerPath = Join-Path $TestOutput 'Wenqu-Setup-0.2.3-win-x64.exe'
    $FirstExecutable = 'Wenqu.exe'
    $InitialInstaller = $InstallerPath
    if ($LegacyBundle) {
        $LegacyRoot = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot $LegacyBundle)).Path
        if (-not (Test-Path -LiteralPath (Join-Path $LegacyRoot 'ObsidianRAG.exe'))) { throw 'Legacy bundle executable missing.' }
        & $Compiler '/Qp' "/DMyAppId=WenquAcceptance-$RunId" '/DTestBuild=1' "/DInstallerOutputDir=$TestOutput" `
            '/DMyAppName=ObsidianRAG' '/DMyAppVersion=0.2.2' '/DMyAppExeName=ObsidianRAG.exe' "/DBundleDir=$LegacyRoot" (Join-Path $ProjectRoot 'installer\Wenqu.iss')
        if ($LASTEXITCODE -ne 0) { throw 'Legacy acceptance installer compilation failed.' }
        $InitialInstaller = Join-Path $TestOutput 'Wenqu-Setup-0.2.2-win-x64.exe'
        $FirstExecutable = 'ObsidianRAG.exe'
    }
} else {
    throw 'Pass no Installer: acceptance must compile an isolated installer with a dedicated AppId.'
}

foreach ($Path in @($InstallRoot, $DataRoot)) {
    $FullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $FullPath.StartsWith($TempBase, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to use a path outside TEMP: $FullPath"
    }
}

$Install = Start-Process -FilePath $InitialInstaller -ArgumentList @(
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
        -Executable (Join-Path $InstallRoot $FirstExecutable) -Port $Port `
        -DataRoot $DataRoot -ExpectedMinimumConversations 0
    $ExpectedDataRetention = $true
    # An old-branded executable in this install must be removed by the upgrade.
    if (-not $LegacyBundle) {
        Copy-Item -LiteralPath (Join-Path $InstallRoot 'Wenqu.exe') -Destination (Join-Path $InstallRoot 'ObsidianRAG.exe')
    }
    $Upgrade = Start-Process -FilePath $InstallerPath -ArgumentList @(
        '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/NOICONS',
        "/LOG=$InstallLog"
    ) -WindowStyle Hidden -PassThru -Wait
    if ($Upgrade.ExitCode -ne 0) { throw "Upgrade install failed: $($Upgrade.ExitCode)" }
    if (Test-Path -LiteralPath (Join-Path $InstallRoot 'ObsidianRAG.exe')) { throw 'Legacy executable was not removed.' }
    & (Join-Path $ProjectRoot 'scripts\test_product_bundle.ps1') `
        -Executable (Join-Path $InstallRoot 'Wenqu.exe') -Port ($Port + 1) `
        -DataRoot $DataRoot -ExpectedMinimumConversations 1 -ExpectFreshImport $false
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
    if (Test-Path -LiteralPath $TestOutput) {
        $VerifiedOutput = [System.IO.Path]::GetFullPath($TestOutput)
        if (-not $VerifiedOutput.StartsWith($TempBase, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe test output path.' }
        Remove-Item -LiteralPath $VerifiedOutput -Recurse -Force
    }
}
