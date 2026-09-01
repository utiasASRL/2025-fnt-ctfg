# Rebuild the standalone code-listing PDFs that chap5.tex includes as figures.
#
# Usage:
#   .\build_listings.ps1            # rebuild every listing
#   .\build_listings.ps1 wnoa       # rebuild only wnoa.tex
#
# Each <name>.tex in this directory is a `standalone` document holding one code
# listing; compiling it produces <name>.pdf, which chap5.tex pulls in with
# \includegraphics.  Auxiliary files are written to build\ to keep this
# directory clean.

param([string[]]$Names)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$buildDir = 'build'
if (-not (Test-Path $buildDir)) { New-Item -ItemType Directory $buildDir | Out-Null }

if ($Names) {
  $targets = $Names | ForEach-Object { [IO.Path]::GetFileNameWithoutExtension($_) }
} else {
  $targets = Get-ChildItem -Filter *.tex |
             Where-Object { $_.Name -ne 'preamble.tex' } |
             ForEach-Object { $_.BaseName }
}

foreach ($name in $targets) {
  if (-not (Test-Path "$name.tex")) { throw "error: $name.tex not found" }
  Write-Host "==> $name.tex"
  # NB: the -output-directory argument must be quoted, otherwise Windows
  # PowerShell passes "$buildDir" through literally.
  & pdflatex -interaction=nonstopmode -halt-on-error "-output-directory=$buildDir" "$name.tex" | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "pdflatex failed on $name.tex (see $buildDir\$name.log)" }
  Move-Item -Force "$buildDir\$name.pdf" "$name.pdf"
  Write-Host "    -> $name.pdf"
}

Write-Host 'Done. Recompile the main document to pick up the new figures.'
