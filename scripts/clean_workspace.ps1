# Interactive cleanup script for sampling_fg workspace
# Usage: Run this script from repository root in PowerShell (may require -ExecutionPolicy Bypass)

Param(
    [switch]$DoIt
)

$root = Resolve-Path .
Write-Host "Repository root: $root"

$items = @(
    @{ path = "outputs"; action = "archive"; desc = "Generated outputs and example results" },
    @{ path = "fg_vector_sampler.egg-info"; action = "archive"; desc = "Packaging metadata" },
    @{ path = "**/__pycache__"; action = "remove"; desc = "Python bytecode caches" },
    @{ path = ".venv"; action = "remove"; desc = "Local virtual environment (large)" },
    @{ path = ".vscode"; action = "remove"; desc = "Editor settings" }
)

$archiveDir = Join-Path $root "archive"
if (!(Test-Path $archiveDir)) { New-Item -ItemType Directory -Path $archiveDir | Out-Null }

Write-Host "The script will consider these changes:`n"
foreach ($it in $items) { Write-Host " - $($it.path): $($it.desc) -> $($it.action)" }

if (-not $DoIt) {
    Write-Host "\nDry run only. To perform actions, re-run with -DoIt switch.\n"
}

foreach ($it in $items) {
    $pattern = $it.path
    $action = $it.action

    if ($pattern -like "**/*") {
        # glob: find matches
        $matches = Get-ChildItem -Path $root -Recurse -Force -Filter "__pycache__" -ErrorAction SilentlyContinue
    } else {
        $matches = @()
        $p = Join-Path $root $pattern
        if (Test-Path $p) { $matches += Get-Item -LiteralPath $p }
    }

    if ($matches.Count -eq 0) { continue }

    foreach ($m in $matches) {
        $src = $m.FullName
        if ($action -eq "archive") {
            $dest = Join-Path $archiveDir $m.Name
            Write-Host "Archive: $src -> $dest"
            if ($DoIt) {
                if (Test-Path $dest) { $dest = Join-Path $archiveDir ("$($m.Name)_$(Get-Date -Format yyyyMMddHHmmss)") }
                Move-Item -LiteralPath $src -Destination $dest -Force
            }
        } elseif ($action -eq "remove") {
            Write-Host "Remove: $src"
            if ($DoIt) {
                Remove-Item -LiteralPath $src -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

Write-Host "Done. If you ran dry-run, re-run with -DoIt to apply changes."
