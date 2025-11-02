# Correct base (SpamAssassin public corpus)
$base = "https://spamassassin.apache.org/old/publiccorpus"
$dst  = "data/raw/spamassassin"

# files you want to fetch
$files = @(
  "20021010_easy_ham.tar.bz2",
  "20021010_easy_ham_2.tar.bz2",
  "20021010_hard_ham.tar.bz2",
  "20021010_spam.tar.bz2",
  "20030228_easy_ham.tar.bz2",
  "20030228_easy_ham_2.tar.bz2",
  "20030228_hard_ham.tar.bz2",
  "20030228_spam_2.tar.bz2"  # adjust list if file names differ
)

# create dest dir
New-Item -ItemType Directory -Force -Path $dst | Out-Null

foreach ($f in $files) {
  $url = "$base/$f"
  $out = Join-Path $dst $f
  $temp = "$out.part"

  Write-Host "Downloading $f ..."
  try {
    # Remove any previous partial file
    if (Test-Path $temp) { Remove-Item $temp -Force }
    # Download to temporary file; Stop on error
    Invoke-WebRequest -Uri $url -OutFile $temp -ErrorAction Stop

    # Verify file exists and has size > 0
    if ((Test-Path $temp) -and ((Get-Item $temp).Length -gt 0)) {
      # atomically rename to final filename
      Move-Item -Force $temp $out

      Write-Host "Downloaded: $out (size $((Get-Item $out).Length) bytes)"

      # Extract using tar. On Windows 10+, tar is available. If not, install bsdtar/7zip.
      # -xjf works for .tar.bz2 (bzip2). If you have GNU tar that supports it.
      try {
        tar -xjf $out -C $dst
        Write-Host "Extracted $f"
      } catch {
        Write-Warning "Extraction failed for $out. Check tar availability or try 7zip."
      }
    } else {
      Write-Warning "Downloaded file appears empty: $temp"
    }
  } catch {
    Write-Warning "Failed to download $url : $($_.Exception.Message)"
    if (Test-Path $temp) { Remove-Item $temp -Force }
  }
}
