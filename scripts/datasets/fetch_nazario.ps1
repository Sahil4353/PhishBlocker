# scripts/datasets/fetch_nazario.ps1
# Download Jose Nazario phishing corpus via explicit filename list.
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File .\scripts\datasets\fetch_nazario.ps1

param(
    [string]$BaseUrl = "https://monkey.org/~jose/phishing/",
    [string]$OutDir = "data/raw/nazario"
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Explicit list based on the index you pasted
$files = @(
    "20051114.mbox",
    "README.txt",
    "phishing-2015",
    "phishing-2016",
    "phishing-2017",
    "phishing-2018",
    "phishing-2019",
    "phishing-2020",
    "phishing-2021",
    "phishing-2022",
    "phishing-2023",
    "phishing-2024",
    "phishing0.mbox",
    "phishing1.mbox",
    "phishing2.mbox",
    "phishing3.mbox",
    "private-phishing4.mbox"
)

Write-Host "Base URL: $BaseUrl"
Write-Host "Output dir: $OutDir"
Write-Host "Will download $($files.Count) files:`n"
$files | ForEach-Object { "  $_" }

function Download-File {
    param(
        [string]$Uri,
        [string]$OutPath,
        [int]$MaxRetries = 3
    )

    $temp = "$OutPath.part"
    $attempt = 0
    while ($attempt -lt $MaxRetries) {
        $attempt++
        try {
            if (Test-Path $temp) {
                Remove-Item $temp -Force -ErrorAction SilentlyContinue
            }
            Write-Host "  [Attempt $attempt] $Uri"
            Invoke-WebRequest -Uri $Uri -OutFile $temp -UseBasicParsing -ErrorAction Stop

            if ((Test-Path $temp) -and ((Get-Item $temp).Length -gt 0)) {
                Move-Item -Force $temp $OutPath
                return $true
            }
            else {
                Write-Warning "  Zero-length download for $Uri"
                if (Test-Path $temp) {
                    Remove-Item $temp -Force -ErrorAction SilentlyContinue
                }
            }
        }
        catch {
            Write-Warning "  Failed to download $Uri : $($_.Exception.Message)"
        }

        Start-Sleep -Seconds (2 * $attempt)
    }

    Write-Error "  Giving up on $Uri after $MaxRetries attempts."
    return $false
}

Write-Host "`nStarting downloads…`n"

foreach ($name in $files) {
    $uri = $BaseUrl + $name
    $outPath = Join-Path $OutDir $name

    if (Test-Path $outPath) {
        Write-Host "Skipping (exists): $name"
        continue
    }

    Write-Host "Downloading: $name"
    $ok = Download-File -Uri $uri -OutPath $outPath
    if (-not $ok) {
        Write-Warning "Failed to download: $name"
    }
}

Write-Host "`nDone. Files saved under: $OutDir`n"
Get-ChildItem -File $OutDir | Format-Table Name, Length
