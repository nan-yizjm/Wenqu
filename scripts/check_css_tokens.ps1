# 门禁：styles.css 里除了 :root / html[data-theme=dark] 两个变量声明块，
# 不允许出现颜色字面量。CSS 重构没有任何单测能发现回归，只能靠这道静态检查。
$ErrorActionPreference = 'Stop'

$Path = Join-Path $PSScriptRoot '..\web\src\styles.css'
$Text = Get-Content $Path -Raw -Encoding UTF8

# 变量声明块是唯一允许写死颜色的地方，先整段摘掉。
$Declarations = [regex]::Matches($Text, ':root\s*\{[^}]*\}|html\[data-theme="dark"\]\s*\{[^}]*\}')
$Body = $Text
foreach ($Match in $Declarations) { $Body = $Body.Replace($Match.Value, '') }

$Literals = [regex]::Matches($Body, '#[0-9a-fA-F]{3,8}(?![0-9a-zA-Z])')
if ($Literals.Count -gt 0) {
    $Lines = $Literals | ForEach-Object { $_.Value } | Sort-Object -Unique
    throw "styles.css 在变量块之外还有 $($Literals.Count) 处颜色字面量：$($Lines -join ', ')"
}

# 变量必须成对定义：浅色缺一个 key，深色就会回落到浅色值。
function Get-Tokens([string] $Block) {
    [regex]::Matches($Block, '(--[a-z0-9-]+)\s*:') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
}

$LightBlock = ($Declarations | Where-Object { $_.Value.StartsWith(':root') }).Value
$DarkBlock = ($Declarations | Where-Object { $_.Value.StartsWith('html[') }).Value
if (-not $LightBlock -or -not $DarkBlock) { throw '没有找到 :root 或 html[data-theme="dark"] 变量块。' }

$LightTokens = Get-Tokens $LightBlock
$DarkTokens = Get-Tokens $DarkBlock
$Missing = $LightTokens | Where-Object { $_ -notin $DarkTokens }
if ($Missing) { throw "深色主题缺少这些变量：$($Missing -join ', ')" }

$Used = [regex]::Matches($Body, 'var\(\s*(--[a-z0-9-]+)') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
$Unused = $LightTokens | Where-Object { $_ -notin $Used }
if ($Unused) { throw "定义了但从未使用的变量：$($Unused -join ', ')" }

Write-Host "styles.css 变量检查通过：$($LightTokens.Count) 个变量，浅色/深色成对，无字面量残留。"