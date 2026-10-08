param([switch]$SkipInstaller)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $ProjectRoot
function Assert-NativeSuccess([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE" }
}

npm --prefix web ci
Assert-NativeSuccess 'npm ci'
npm --prefix web run build
Assert-NativeSuccess 'web build'

if (-not (Test-Path .venv-product\Scripts\python.exe)) {
    $Python312 = py -0p | Where-Object { $_ -match '3\.12' } | ForEach-Object {
        ($_ -replace '^\s*-\S+\s+', '').Trim()
    } | Select-Object -First 1
    if (-not $Python312 -or -not (Test-Path -LiteralPath $Python312)) {
        throw 'Python 3.12 x64 is required to build the Windows package.'
    }
    & $Python312 -m venv .venv-product
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the Python 3.12 product environment.' }
}
.\.venv-product\Scripts\python.exe -m pip install --upgrade pip
Assert-NativeSuccess 'pip upgrade'
.\.venv-product\Scripts\python.exe -m pip install `
    'torch==2.13.0' --index-url https://download.pytorch.org/whl/cpu
Assert-NativeSuccess 'CPU PyTorch install'
.\.venv-product\Scripts\python.exe -m pip install `
    -r requirements-product.txt -r requirements-build.txt
Assert-NativeSuccess 'product dependencies install'
.\.venv-product\Scripts\python.exe .\scripts\generate_license_inventory.py
Assert-NativeSuccess 'license inventory'
.\.venv-product\Scripts\python.exe -m PyInstaller `
    --noconfirm --clean product.spec
Assert-NativeSuccess 'PyInstaller'

if (-not $SkipInstaller) {
    $Compiler = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if (-not $Compiler) {
        $Candidates = @(
            (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe'),
            'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
        )
        $Candidate = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
        if ($Candidate) { $Compiler = Get-Item -LiteralPath $Candidate }
    }
    if (-not $Compiler) { throw 'Inno Setup 6 was not found. The onedir bundle was built, but the installer was not.' }
    $CompilerPath = if ($Compiler -is [System.Management.Automation.CommandInfo]) {
        $Compiler.Source
    } else {
        $Compiler.FullName
    }
    & $CompilerPath '/Qp' .\installer\Wenqu.iss
    Assert-NativeSuccess 'Inno Setup'
    $Installer = Resolve-Path '.\dist-installer\Wenqu-Setup-0.2.3-win-x64.exe'
    $Hash = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant()
    [System.IO.File]::WriteAllText(
        "$Installer.sha256", "$Hash  $([System.IO.Path]::GetFileName($Installer))`n",
        [System.Text.UTF8Encoding]::new($false))
}
