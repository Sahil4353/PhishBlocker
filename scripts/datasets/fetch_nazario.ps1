# PowerShell downloader for Nazario phishing corpus (one-level)
# Usage: run in your project folder. Adjust $dst if you want another path.

$base = "https://monkey.org/~jose/phishing/"
$dst = "data/raw/nazario"
New-Item -ItemType Directory -Force -Path $dst | Out-Null

function Download-File {
    param($uri, $outPath, $maxRetries = 2)
    $temp = "$outPath.part"
    $attempt = 0
    while ($attempt -le $maxRetries) {
        try {
            if (Test-Path $temp) { Remove-Item $temp -Force -ErrorAction SilentlyContinue }
            Invoke-WebRequest -Uri $uri -OutFile $temp -ErrorAction Stop -UseBasicParsing
            if ((Test-Path $temp) -and ((Get-Item $temp).Length -gt 0)) {
                Move-Item -Force $temp $outPath
                return $true
            }
            else {
                Remove-Item $temp -Force -ErrorAction SilentlyContinue
                throw "Zero-length download"
            }
        }
        catch {
            $attempt++
            Write-Warning "Attempt $attempt failed for $uri. $($_.Exception.Message)"
            Start-Sleep -Seconds (2 * $attempt)
            if ($attempt -gt $maxRetries) { return $false }
        }
    }
}

Write-Host "Fetching directory index from $base ..."
$response = Invoke-WebRequest -Uri $base -UseBasicParsing

# collect top-level file links (.eml, .mbox, .txt)
$topFiles = $response.Links | Where-Object { $_.href -match '\.(eml|mbox|txt)$' } | Select-Object -Unique
if ($topFiles.Count -eq 0) { Write-Warning "No top-level files found in index. Check $base in a browser." }

foreach ($l in $topFiles) {
    $url = [System.Uri]::new($base, $l.href).AbsoluteUri
    $name = Split-Path $url -Leaf
    $out = Join-Path $dst $name
    if (-not (Test-Path $out)) {
        Write-Host "Downloading top-level: $name"
        if (-not (Download-File -uri $url -outPath $out)) {
            Write-Warning "Failed to download $url"
        }
    }
    else {
        Write-Host "Skipping (exists): $name"
    }
}

# Find one-level subdirectories and fetch .eml/.txt from them
$subdirs = $response.Links | Where-Object { $_.href -match '/$' -and $_.href -ne "../" } | Select-Object -Unique
foreach ($d in $subdirs) {
    $subUrl = [System.Uri]::new($base, $d.href).AbsoluteUri
    Write-Host "Scanning subdir: $subUrl"
    try {
        $subResp = Invoke-WebRequest -Uri $subUrl -UseBasicParsing -ErrorAction Stop
        $emls = $subResp.Links | Where-Object { $_.href -match '\.(eml|txt|mbox)$' } | Select-Object -Unique
        foreach ($e in $emls) {
            $eu = [System.Uri]::new($subUrl, $e.href).AbsoluteUri
            # prefix file with subdir name to avoid collisions
            $prefix = (Split-Path $d.href.TrimEnd('/') -Leaf)
            $name = $prefix + "_" + (Split-Path $eu -Leaf)
            $out = Join-Path $dst $name
            if (-not (Test-Path $out)) {
                Write-Host "  Downloading: $name"
                if (-not (Download-File -uri $eu -outPath $out)) {
                    Write-Warning "  Failed to download $eu"
                }
            }
            else {
                Write-Host "  Skipping (exists): $name"
            }
        }
    }
    catch {
        Write-Warning "Failed to read subdir $subUrl : $($_.Exception.Message)"
    }
}

Write-Host "Done. Files saved under: $dst"
