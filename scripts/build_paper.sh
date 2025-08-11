#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../paper"
mkdir -p build
if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk not found. Please install TeX Live/MacTeX (includes latexmk)." >&2
  exit 1
fi
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex | cat
cp build/main.pdf ./main.pdf
echo "Paper built: paper/main.pdf"

