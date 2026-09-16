param(
    [string]$Executable = ".\dist\ObsidianRAG\ObsidianRAG.exe",
    [int]$Port = 8876,
    [string]$DataRoot = "",
    [int]$ExpectedMinimumConversations = 0,
    [bool]$ExpectFreshImport = $true
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ExecutableCandidate = if ([System.IO.Path]::IsPathRooted($Executable)) {
    $Executable
} else {
    Join-Path $ProjectRoot $Executable
}
$ExecutablePath = (Resolve-Path -LiteralPath $ExecutableCandidate).Path
$OwnsTestRoot = [string]::IsNullOrWhiteSpace($DataRoot)
$TestRoot = if ($OwnsTestRoot) {
    Join-Path ([System.IO.Path]::GetTempPath()) ("ObsidianRAG-bundle-" + [guid]::NewGuid())
} else { $DataRoot }
New-Item -ItemType Directory -Path $TestRoot -Force | Out-Null
$ResolvedTestRoot = (Resolve-Path -LiteralPath $TestRoot).Path
$ExpectedTempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
if (-not $ResolvedTestRoot.StartsWith($ExpectedTempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use a test path outside TEMP: $ResolvedTestRoot"
}

# The server answers with `application/json` and no charset parameter. Faced with
# that, Windows PowerShell 5.1 decodes the body as ISO-8859-1 and turns every
# UTF-8 byte into its own character, so a bundled file name arrives as 15
# Latin-1 characters instead of "welcome.md" in Chinese. That is a different
# problem from the script encoding fixed below (line ~70) and it breaks the
# name comparisons just as badly. Decode the raw bytes as UTF-8 explicitly so
# Windows PowerShell 5.1 and PowerShell 7 agree.
function Get-JsonUtf8 {
    param([Parameter(Mandatory)][string]$Uri)
    $Client = New-Object System.Net.WebClient
    $Client.Encoding = [System.Text.Encoding]::UTF8
    try { return ($Client.DownloadString($Uri) | ConvertFrom-Json) }
    finally { $Client.Dispose() }
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
    $Existing = Invoke-RestMethod "http://127.0.0.1:$Port/api/v1/conversations"
    if ($Existing.conversations.Count -lt $ExpectedMinimumConversations) {
        throw "Expected at least $ExpectedMinimumConversations preserved conversations."
    }
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
    # Windows PowerShell 5.1 reads UTF-8 scripts without a BOM using the local
    # code page. Construct Chinese resource names from code points so this
    # release check behaves identically in Windows PowerShell and PowerShell 7.
    $GuideName = (-join ([char[]](0x7528, 0x6237, 0x6307, 0x5357))) + '.md'
    $WelcomeName = (-join ([char[]](0x6b22, 0x8fce, 0x4f7f, 0x7528))) + '.md'
    $Resources = Get-JsonUtf8 "http://127.0.0.1:$Port/api/v1/resources"
    if ($Resources.docs.Count -lt 1) { throw 'The packaged bundle listed no bundled documents.' }
    if (-not ($Resources.examples | Where-Object { $_.name -eq $WelcomeName })) {
        throw 'The packaged bundle did not list the welcome example.'
    }
    # Escape the example name to \uXXXX so the request body stays ASCII; Windows
    # PowerShell 5.1 would otherwise encode the literal Chinese via the local
    # code page and the server would reject the name.
    $WelcomeEscaped = -join ($WelcomeName.ToCharArray() | ForEach-Object { '\u' + ([int]$_).ToString('x4') })
    $Import = Invoke-RestMethod "http://127.0.0.1:$Port/api/v1/resources/import" `
        -Method Post -ContentType 'application/json; charset=utf-8' `
        -Body ('{"name":"' + $WelcomeEscaped + '"}')
    # The installer test calls this check twice against the SAME data root: once
    # before the upgrade install, where the example must be new, and once after
    # it, where the same example must have survived and is therefore reported as
    # already imported. Assert both directions instead of only the first one.
    $ExpectedAlreadyImported = -not $ExpectFreshImport
    if ($Import.already_imported -ne $ExpectedAlreadyImported) {
        throw "Bundled example import reported already_imported=$($Import.already_imported), expected $ExpectedAlreadyImported. Data retention across the install may have broken."
    }
    $Indexed = $false
    for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
        $Documents = Get-JsonUtf8 "http://127.0.0.1:$Port/api/v1/documents"
        $Example = $Documents.documents | Where-Object { $_.display_name -eq $WelcomeName }
        if ($Example -and $Example.status -eq 'ready') { $Indexed = $true; break }
        if ($Example -and $Example.status -eq 'failed') { throw "Bundled example import failed: $($Example.error)" }
        Start-Sleep -Milliseconds 500
    }
    if (-not $Indexed) { throw 'The bundled example was not searchable within 30 seconds.' }
    $Search = Get-JsonUtf8 "http://127.0.0.1:$Port/api/v1/search?q=PagedAttention"
    if ($Search.results.Count -lt 1) { throw 'Searching PagedAttention found no hit in the bundled example.' }
    $Reimport = Invoke-RestMethod "http://127.0.0.1:$Port/api/v1/resources/import" `
        -Method Post -ContentType 'application/json; charset=utf-8' `
        -Body ('{"name":"' + $WelcomeEscaped + '"}')
    if (-not $Reimport.already_imported) { throw 'Re-importing the same bundled example was not idempotent.' }
    Invoke-RestMethod "http://127.0.0.1:$Port/api/v1/system/shutdown" -Method Post | Out-Null
    if (-not $Process.WaitForExit(10000)) { throw 'Product did not exit within 10 seconds.' }
    $BundleRoot = Split-Path -Parent $ExecutablePath
    foreach ($Relative in @((Join-Path '_internal\resources\docs' $GuideName),
            '_internal\resources\docs\THIRD_PARTY_LICENSES.md',
            (Join-Path '_internal\resources\examples' $WelcomeName))) {
        if (-not (Test-Path -LiteralPath (Join-Path $BundleRoot $Relative))) {
            throw "Packaged resource is missing: $Relative"
        }
    }
    Write-Host "Bundle smoke test passed: schema=$($Health.database_schema), port=$Port, preserved=$($Existing.conversations.Count)"
} finally {
    if ($Process -and -not $Process.HasExited) { Stop-Process -Id $Process.Id -Force }
    if ($OwnsTestRoot) { Remove-Item -LiteralPath $ResolvedTestRoot -Recurse -Force }
}
