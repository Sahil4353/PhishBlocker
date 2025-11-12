# Compact project tree view (run from repo root)
# Shows up to 4 levels deep and 2 files per directory

$excludePatterns = @(
    '\.git',
    '\.venv',
    'data\\raw\\enron\\maildir',
    'data\\raw\\spamassassin',
    '__pycache__'
)

function Show-Tree {
    param(
        [string]$Path = ".",
        [int]$Level = 0,
        [int]$MaxDepth = 4,
        [int]$MaxFilesPerDir = 2
    )

    if ($Level -gt $MaxDepth) { return }

    $fullPath = (Resolve-Path $Path).Path

    # Skip excluded paths
    foreach ($pattern in $excludePatterns) {
        if ($fullPath -like "*$pattern*") { return }
    }

    $indent = ' ' * ($Level * 2)

    if ($Level -eq 0) {
        Write-Host (Split-Path $fullPath -Leaf)
    }

    # Show a couple of files in this directory
    $files = Get-ChildItem $fullPath -File -ErrorAction SilentlyContinue |
    Select-Object -First $MaxFilesPerDir

    foreach ($f in $files) {
        Write-Host "$indent- $($f.Name)"
    }

    # Recurse into subdirectories
    $dirs = Get-ChildItem $fullPath -Directory -ErrorAction SilentlyContinue
    foreach ($d in $dirs) {
        $dPath = $d.FullName

        $excluded = $false
        foreach ($pattern in $excludePatterns) {
            if ($dPath -like "*$pattern*") { $excluded = $true; break }
        }
        if ($excluded) { continue }

        Write-Host "$indent+ $($d.Name)"
        Show-Tree -Path $dPath -Level ($Level + 1) -MaxDepth $MaxDepth -MaxFilesPerDir $MaxFilesPerDir
    }
}

Show-Tree

# Replace with a suspect folder you saw, e.g. the allen-p folder
$folder = ".\data\raw\enron\maildir\allen-p\all_documents"
if (-not (Test-Path $folder)) { Write-Host "Folder not found: $folder"; exit }

# Show first 20 files with size and LastWriteTime
Get-ChildItem -LiteralPath $folder -File -Recurse -ErrorAction SilentlyContinue |
Sort-Object Length -Descending |
Select-Object -First 20 FullName, Length, LastWriteTime |
Format-Table -AutoSize

$folder = ".\data\raw\enron\maildir\allen-p\all_documents"
$sample = Get-ChildItem -LiteralPath $folder -File -Recurse -ErrorAction SilentlyContinue |
          Select-Object -First 1

$sample.FullName

$path = $sample.FullName
$extPath = "\\?\$path"   # extended-length path that preserves trailing dot

Get-Content -LiteralPath $extPath -TotalCount 40

    