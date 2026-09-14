param([switch]$SkipInstaller)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $ProjectRoot

npm --prefix web ci
npm --prefix web run build

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
.\.venv-product\Scripts\python.exe -m pip install `
    'torch==2.13.0' --index-url https://download.pytorch.org/whl/cpu
.\.venv-product\Scripts\python.exe -m pip install `
    -r requirements-product.txt -r requirements-build.txt
.\.venv-product\Scripts\python.exe .\scripts\generate_license_inventory.py
.\.venv-product\Scripts\python.exe -m PyInstaller `
    --noconfirm --clean product.spec

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
    & $CompilerPath .\installer\ObsidianRAG.iss
    $Installer = Resolve-Path '.\dist-installer\ObsidianRAG-Setup-0.2.0-win-x64.exe'
    $Hash = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant()
    [System.IO.File]::WriteAllText(
        "$Installer.sha256", "$Hash  $([System.IO.Path]::GetFileName($Installer))`n",
        [System.Text.UTF8Encoding]::new($false))
}
