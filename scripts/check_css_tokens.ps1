# 门禁：styles.css 的 token 块之外不允许出现颜色字面量，且六块（默认浅色 + 默认深色
# + 3 套模板 x 浅/深）的键名必须完全一致——少一个键，切到那套皮肤时该属性会串回上
# 一套的值，界面上只是"颜色有点怪"，几乎不可能被肉眼定位。CSS 重构没有任何单测能
# 发现回归，只能靠这道静态检查。
$ErrorActionPreference = 'Stop'

$Path = Join-Path $PSScriptRoot '..\web\src\styles.css'
$Text = Get-Content $Path -Raw -Encoding UTF8

# 变量声明块是唯一允许写死颜色的地方，先整段摘掉。
$BlockPattern = '(?::root\s*\{[^}]*\}|html\[data-theme="dark"\]\s*\{[^}]*\}|html\[data-template="[a-z]+"\]\s*\{[^}]*\}|html\[data-template="[a-z]+"\]\[data-theme="dark"\]\s*\{[^}]*\})'
$Declarations = [regex]::Matches($Text, $BlockPattern)
if ($Declarations.Count -ne 6) {
    throw "token 块应有 6 个（默认浅色 + 默认深色 + 3 套模板 x 浅/深），实际 $($Declarations.Count) 个。"
}
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

$Signatures = @($Declarations | ForEach-Object { (Get-Tokens $_.Value) -join ',' })
if (($Signatures | Sort-Object -Unique).Count -ne 1) {
    $Sizes = @($Declarations | ForEach-Object { (Get-Tokens $_.Value).Count })
    throw "六块 token 的键名不一致（各块键数：$($Sizes -join ', ')）——切模板会串色。"
}

$LightBlock = ($Declarations | Where-Object { $_.Value.StartsWith(':root') }).Value
$LightTokens = Get-Tokens $LightBlock

$Used = [regex]::Matches($Body, 'var\(\s*(--[a-z0-9-]+)') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
$Unused = $LightTokens | Where-Object { $_ -notin $Used }
if ($Unused) { throw "定义了但从未使用的变量：$($Unused -join ', ')" }

Write-Host "styles.css 变量检查通过：$($LightTokens.Count) 个变量，六块键名一致，无字面量残留。"