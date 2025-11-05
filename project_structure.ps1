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
