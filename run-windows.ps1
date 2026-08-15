[CmdletBinding()]
param(
    [string]$Addr = "0.0.0.0:3000",
    [string]$Hotwords = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ModelRevision = "b485a2da4b96087df52319c40a81f95329951a81"
$ModelBase = "https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-tract-offline/resolve/$ModelRevision"
$ModelPath = Join-Path $PSScriptRoot "models/model.onnx"
$TokensPath = Join-Path $PSScriptRoot "models/tokens.txt"

function Get-PinnedFile {
    param(
        [string]$Url,
        [string]$Path,
        [string]$Sha256,
        [long]$Size
    )

    $valid = Test-Path -LiteralPath $Path
    if ($valid) {
        $file = Get-Item -LiteralPath $Path
        $valid = $file.Length -eq $Size -and (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Sha256
    }
    if ($valid) {
        Write-Host "[shenava] verified $Path"
        return
    }

    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $partial = "$Path.partial"
    Write-Host "[shenava] downloading pinned asset $Url"
    Invoke-WebRequest -Uri $Url -OutFile $partial
    $download = Get-Item -LiteralPath $partial
    $hash = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash
    if ($download.Length -ne $Size -or $hash -ne $Sha256) {
        Remove-Item -LiteralPath $partial -Force
        throw "Downloaded asset failed size/SHA-256 verification: $Path"
    }
    Move-Item -LiteralPath $partial -Destination $Path -Force
}

Get-PinnedFile -Url "$ModelBase/model.onnx" -Path $ModelPath `
    -Sha256 "0BFDF9FC3C531F351AD02D7D6B4B309DA7FF2F73EB20E2AE167FA12D074C75AC" `
    -Size 437718188
Get-PinnedFile -Url "$ModelBase/tokens.txt" -Path $TokensPath `
    -Sha256 "8E192963F6E666DFA5721E5CBD4710BC1EF592460A45F08CEFC94B2DB16A6954" `
    -Size 12236

Push-Location $PSScriptRoot
try {
    cargo build --release --locked --no-default-features --features cpu-only
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }

    $serverArgs = @(
        "--model", $ModelPath,
        "--tokens", $TokensPath,
        "--mel", (Join-Path $PSScriptRoot "assets/mel_filters.json"),
        "--backend", "cpu",
        "--addr", $Addr
    )
    if ($Hotwords) {
        $serverArgs += @("--hotwords", (Resolve-Path -LiteralPath $Hotwords).Path)
    }
    & (Join-Path $PSScriptRoot "target/release/shenava-asr-server.exe") @serverArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
