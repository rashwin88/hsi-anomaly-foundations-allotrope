# One-shot documentation cleanup - 2026-08-24
#
# Removes the documentation superseded by docs/01-orientation.md ... docs/09-known-issues.md
# Everything deleted here is recoverable:  git checkout 8094a50 -- <path>
#
# Run from the repo root:   powershell -ExecutionPolicy Bypass -File scripts\cleanup_docs.ps1
# Preview without deleting: powershell -ExecutionPolicy Bypass -File scripts\cleanup_docs.ps1 -WhatIf
#
# NOTE: keep this file pure ASCII. Windows PowerShell 5.1 reads .ps1 as ANSI,
# so non-ASCII characters (em dashes, smart quotes) produce parse errors.

[CmdletBinding(SupportsShouldProcess = $true)]
param()

$ErrorActionPreference = "Stop"

if (-not (Test-Path "docs\01-orientation.md")) {
    throw "Run this from the repo root - docs/01-orientation.md not found."
}

$before = (Get-ChildItem -Recurse -File -Filter *.md |
           Where-Object { $_.FullName -notmatch 'node_modules' }).Count

# --- Root files: stale, superseded by README.md + docs/01-09 -----------------
# CONTEXT.md   phase-1 only; omits foundation_models, backend, frontend
# TOC.md       index that never listed docs/tech
# TODO.md      20 of 22 done; open items were AWS procurement + a GMM writeup
# TESTIN.md / running_tests.md  -> docs/02-setup.md
$root = @("CONTEXT.md", "TOC.md", "TODO.md", "TESTIN.md", "running_tests.md")

# --- docs/tech/: 130 files ---------------------------------------------------
# Chapter 1 documented classes and enum members that do not exist; six files in
# chapter 5 documented a pixel_stats_override mechanism that was never implemented;
# every source link in the tree was broken. Consolidated into docs/03, 04, 05.
$tech = @("docs\tech")

# --- Legacy docs/<topic>/ folders --------------------------------------------
# Superseded by docs/03-05, or empty / half-written stubs.
$legacy = @(
    "docs\anomaly_detection", "docs\concepts",            "docs\data_packaging",
    "docs\dataset_builders",  "docs\design_decisions",    "docs\detectors",
    "docs\drawio",            "docs\experiments",         "docs\explorations",
    "docs\file_handling",     "docs\final_documentation", "docs\image_transformations",
    "docs\model_designs",     "docs\notes",               "docs\paper_reads",
    "docs\patch_generation",  "docs\runbooks",            "docs\seg_former_architecture",
    "docs\stac",              "docs\visualizations"
)

# --- Loose docs/ files -------------------------------------------------------
$loose = @(
    "docs\landsat_intermediate_sharder.md",     # -> docs/03
    "docs\spectral_band_filtering_report.md",   # -> docs/03
    "docs\REMOTE_DEPLOY.md",                    # -> docs/08
    "docs\REMOTE_DEPLOY_QUICKSTART.md",         # -> docs/08
    "docs\Untitled.drawio",
    "monitoring\monitoring_docs.md"             # -> docs/08
)

# --- final design/: raw ideation logs and a point-in-time audit ---------------
# The -spec.md distillations are what the frontend actually cites; these are the
# superseded long-form originals. investigation-* is a 2026-05-11 audit -> docs/09.
$finalDesign = @(
    "final design\abstractions.md",             # -> abstractions-spec.md
    "final design\storyboard.md",               # -> storyboard-spec.md
    "final design\bootstrap.md",                # -> docs/02
    "final design\visuals-and-caching.md",      # -> docs/03 + docs/07
    "final design\investigation-report.md",
    "final design\investigation-spec.md"
)

$files = $root + $loose + $finalDesign
$dirs  = $tech + $legacy

foreach ($f in $files) {
    if (Test-Path $f) {
        if ($PSCmdlet.ShouldProcess($f, "Delete file")) {
            Remove-Item -LiteralPath $f -Force
            Write-Host "  deleted  $f"
        }
    }
}

foreach ($d in $dirs) {
    if (Test-Path $d) {
        $n = (Get-ChildItem -Recurse -File -LiteralPath $d).Count
        $label = "Delete directory, " + $n + " files"
        if ($PSCmdlet.ShouldProcess($d, $label)) {
            Remove-Item -LiteralPath $d -Recurse -Force
            Write-Host ("  deleted  " + $d + "\ (" + $n + " files)")
        }
    }
}

$after = (Get-ChildItem -Recurse -File -Filter *.md |
          Where-Object { $_.FullName -notmatch 'node_modules' }).Count

Write-Host ""
Write-Host ("markdown files: " + $before + " -> " + $after)
Write-Host "Recover anything with:  git checkout 8094a50 -- <path>"
