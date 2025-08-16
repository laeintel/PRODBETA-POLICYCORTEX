Param(
  [Parameter(Mandatory=$true)][string]$PatentFolder
)

function Ensure-Tool($cmd, $installHint) {
  if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
    Write-Host "Missing $cmd. $installHint" -ForegroundColor Yellow
  }
}

Ensure-Tool pandoc "Install from https://pandoc.org/installing.html"
Ensure-Tool mmdc "npm i -g @mermaid-js/mermaid-cli"

$drawings = Join-Path $PatentFolder 'drawings'
$flowcharts = Join-Path $PatentFolder 'flowcharts'
New-Item -ItemType Directory -Force -Path (Join-Path $PatentFolder 'out') | Out-Null
$out = Join-Path $PatentFolder 'out'

Get-ChildItem -Path $drawings -Filter *.mmd -ErrorAction SilentlyContinue | ForEach-Object {
  $svg = Join-Path $out ($_.BaseName + '.svg')
  mmdc -i $_.FullName -o $svg --scale 1 --backgroundColor '#ffffff'
}

Get-ChildItem -Path $flowcharts -Filter *.mmd -ErrorAction SilentlyContinue | ForEach-Object {
  $svg = Join-Path $out ($_.BaseName + '.svg')
  mmdc -i $_.FullName -o $svg --scale 1 --backgroundColor '#ffffff'
}

$mds = @('abstract.md','claims.md','specification.md','figure_list.md','prior_art_search.md') | ForEach-Object { Join-Path $PatentFolder $_ }
$pdf = Join-Path $out 'specification.pdf'

pandoc $mds -o $pdf --from=gfm --metadata title:"Patent Package" --toc --pdf-engine=xelatex

Write-Host "Built: $pdf"

