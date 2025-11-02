# enron_download.ps1
# Downloads canonical Enron archive and extracts it.
# Usage: run from repo root: pwsh .\enron_download.ps1

$Url = "http://www.cs.cmu.edu/~enron/enron_mail_20110402.tgz"   # canonical CMU URL. See CMU hosting. :contentReference[oaicite:1]{index=1}
$DstDir = "data/raw/enron"
$Archive = Join-Path $DstDir (Split-Path $Url -Leaf)
$Temp = "$Archive.part"
$MaxRetries = 4
$SleepBase = 2

New-Item -ItemType Directory -Force -Path $DstDir | Out-Null

function Download-WithRetries {
    param($url, $outPath, $tempPath, $maxRetries)

    $attempt = 0
    while ($attempt -lt $maxRetries) {
        try {
            if (Test-Path $tempPath) { Remove-Item $tempPath -Force -ErrorAction SilentlyContinue }
            Write-Host "Downloading (attempt $($attempt+1)) : $url"
            # Use -UseBasicParsing for compatibility with older PS versions
            Invoke-WebRequest -Uri $url -OutFile $tempPath -UseBasicParsing -ErrorAction Stop

            if ((Test-Path $tempPath) -and ((Get-Item $tempPath).Length -gt 0)) {
                # Optional: compare Content-Length header if available
                try {
                    $head = Invoke-WebRequest -Method Head -Uri $url -UseBasicParsing -ErrorAction SilentlyContinue
                    if ($head -and $head.Headers['Content-Length']) {
                        $remoteLen = [int64]$head.Headers['Content-Length']
                        $localLen = (Get-Item $tempPath).Length
                        if ($localLen -lt $remoteLen) {
                            Write-Warning "Downloaded file size ($localLen) < expected ($remoteLen). Retrying..."
                            throw "Partial download"
                        }
                    }
                }
                catch {
                    # ignore; proceed to rename if file exists
                }

                Move-Item -Force $tempPath $outPath
                Write-Host "Downloaded: $outPath (size $((Get-Item $outPath).Length) bytes)"
                return $true
            }
            else {
                Remove-Item $tempPath -Force -ErrorAction SilentlyContinue
                throw "Zero-length download"
            }
        }
        catch {
            $attempt++
            Write-Warning "Attempt $attempt failed: $($_.Exception.Message)"
            Start-Sleep -Seconds ($SleepBase * $attempt)
            if ($attempt -ge $maxRetries) {
                Write-Error "Download failed after $maxRetries attempts."
                return $false
            }
        }
    }
}

if (-not (Test-Path $Archive)) {
    $ok = Download-WithRetries -url $Url -outPath $Archive -tempPath $Temp -maxRetries $MaxRetries
    if (-not $ok) { throw "Failed to download Enron archive. Check connectivity or open $Url in browser." }
}
else {
    Write-Host "Archive already exists: $Archive"
}

# Extract (tgz = tar.gz)
try {
    Write-Host "Extracting archive to $DstDir ..."
    # tar -xzf works on Windows 10/11 native tar. If your tar doesn't support -z, use 7zip or WSL.
    tar -xzf $Archive -C $DstDir
    Write-Host "Extraction complete."
}
catch {
    Write-Warning "Extraction failed: $($_.Exception.Message)"
    Write-Host "If tar is not available on your system, install 7-Zip or use WSL; or manually extract the archive."
}

# Print where maildir lives (common path)
$expected = Join-Path $DstDir "enron_mail_20110402\maildir"
if (Test-Path $expected) {
    Write-Host "Enron maildir path: $expected"
}
else {
    Write-Host "Extraction done — check $DstDir. Typical maildir path: enron_mail_20110402/maildir/"
}
