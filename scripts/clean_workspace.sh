#!/usr/bin/env bash
# Interactive cleanup script for sampling_fg workspace (POSIX)
set -euo pipefail
root="$(pwd)"
archive_dir="$root/archive"
mkdir -p "$archive_dir"

echo "Repository root: $root"
echo "This script will:"
echo " - move 'outputs' and 'fg_vector_sampler.egg-info' into archive/"
echo " - remove __pycache__ directories"
echo " - (it will NOT remove .venv by default)"

read -p "Proceed with dry-run listing? (y/N) " yn
if [[ "$yn" != "y" && "$yn" != "Y" ]]; then
    echo "Aborting."
    exit 0
fi

# Dry-run listing
if [ -d "outputs" ]; then echo "Would archive: outputs -> $archive_dir/"; fi
if [ -d "fg_vector_sampler.egg-info" ]; then echo "Would archive: fg_vector_sampler.egg-info -> $archive_dir/"; fi

find . -type d -name "__pycache__" -print -exec echo "Would remove: {}" \;

read -p "Apply changes now? (y/N) " yn2
if [[ "$yn2" != "y" && "$yn2" != "Y" ]]; then
    echo "No changes made."; exit 0
fi

# Apply changes
if [ -d "outputs" ]; then mv outputs "$archive_dir/outputs_$(date +%Y%m%d%H%M%S)"; fi
if [ -d "fg_vector_sampler.egg-info" ]; then mv fg_vector_sampler.egg-info "$archive_dir/fg_vector_sampler.egg-info_$(date +%Y%m%d%H%M%S)"; fi

find . -type d -name "__pycache__" -print0 | xargs -0 rm -rf --

echo "Done. Archived into $archive_dir" 
