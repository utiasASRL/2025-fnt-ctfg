#!/usr/bin/env bash
# Rebuild the standalone code-listing PDFs that chap5.tex includes as figures.
#
# Usage:
#   ./build_listings.sh            # rebuild every listing
#   ./build_listings.sh wnoa       # rebuild only wnoa.tex
#
# Each <name>.tex in this directory is a `standalone` document holding one code
# listing; compiling it produces <name>.pdf, which chap5.tex pulls in with
# \includegraphics.  Auxiliary files are written to build/ to keep this
# directory clean.

set -euo pipefail

cd "$(dirname "$0")"

BUILD_DIR=build
mkdir -p "$BUILD_DIR"

if [ "$#" -gt 0 ]; then
  targets=("$@")
else
  targets=()
  for f in *.tex; do
    [ "$f" = "preamble.tex" ] && continue
    targets+=("${f%.tex}")
  done
fi

for name in "${targets[@]}"; do
  name="${name%.tex}"
  if [ ! -f "$name.tex" ]; then
    echo "error: $name.tex not found" >&2
    exit 1
  fi
  echo "==> $name.tex"
  pdflatex -interaction=nonstopmode -halt-on-error \
           -output-directory="$BUILD_DIR" "$name.tex" > /dev/null
  mv "$BUILD_DIR/$name.pdf" "$name.pdf"
  echo "    -> $name.pdf"
done

echo "Done. Recompile the main document to pick up the new figures."
