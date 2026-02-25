# update_repo.ps1
# ============================================================
# KAVAH v1.0-cris-native - Full Repo Update + Tag
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  KAVAH v1.0-cris-native - Repo Update" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$RepoRoot = Get-Location
Set-Location $RepoRoot

# STEP 1: Verify git repo
Write-Host "[1/8] Verifying git repo..." -ForegroundColor Yellow
if (-not (Test-Path ".git")) {
    Write-Host "  ERROR: Not a git repo. Run: git init" -ForegroundColor Red
    exit 1
}

$branch = git rev-parse --abbrev-ref HEAD 2>$null
Write-Host "  OK - branch: $branch" -ForegroundColor Green

# STEP 2: Verify critical source files
Write-Host "[2/8] Checking source files..." -ForegroundColor Yellow

$critical = @(
    "kavah_llm_bridge.py",
    "kavah_reasoner.py",
    "bedrock_agi/main.py",
    "bedrock_agi/core/hbl_geometry.py",
    "bedrock_agi/core/e_model.py",
    "bedrock_agi/core/r_projector.py",
    "bedrock_agi/core/r_invariants.py",
    "bedrock_agi/core/cris_monitor.py",
    "bedrock_agi/core/world_model.py",
    "bedrock_agi/core/riemannian_optimizer.py",
    "bedrock_agi/action/tools/web_search.py",
    "GeometryOfPersistence.v",
    "GeometryOfPersistence.lean"
)

$allPresent = $true
foreach ($f in $critical) {
    if (Test-Path $f) {
        Write-Host "  OK $f" -ForegroundColor Green
    } else {
        Write-Host "  MISSING: $f" -ForegroundColor Red
        $allPresent = $false
    }
}

if (-not $allPresent) {
    Write-Host ""
    Write-Host "  Some files are missing. Continuing with available files." -ForegroundColor Yellow
}

# STEP 3: Backup README
Write-Host "[3/8] Backing up README..." -ForegroundColor Yellow
if ((Test-Path "README.md") -and (-not (Test-Path "README_v0.md"))) {
    Copy-Item "README.md" "README_v0.md" -Force
    Write-Host "  Saved as README_v0.md" -ForegroundColor Green
} elseif (Test-Path "README_v0.md") {
    Write-Host "  README_v0.md already exists - skipping backup" -ForegroundColor Gray
} else {
    Write-Host "  No README.md to back up - skipping" -ForegroundColor Gray
}

# STEP 4: Verify README
Write-Host "[4/8] Checking README.md..." -ForegroundColor Yellow
if (-not (Test-Path "README.md")) {
    Write-Host "  ERROR: README.md not found." -ForegroundColor Red
    Write-Host "  Place README.md in repo root and retry." -ForegroundColor Yellow
    exit 1
}
$readmeSize = (Get-Item "README.md").Length
Write-Host "  README.md present ($readmeSize bytes)" -ForegroundColor Green

# STEP 5: Documentation artifact (lightweight)
Write-Host "[5/8] Writing documentation artifacts..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "research" | Out-Null

@"
# Documentation Update
Repository documentation generated on $(Get-Date).

Artifacts omitted for lightweight commit.
"@ | Out-File -FilePath "research/README_NOTES.md" -Encoding UTF8

Write-Host "  research/README_NOTES.md written" -ForegroundColor Green

# STEP 6: Stage changes
Write-Host "[6/8] Staging changes..." -ForegroundColor Yellow
git add -A

$staged = git diff --cached --name-only
Write-Host "  Staged files:" -ForegroundColor Green
$staged | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }

# STEP 7: Commit
Write-Host "[7/8] Committing..." -ForegroundColor Yellow

$commitMsg = @"
feat: v1.0-cris-native update

- documentation update
- source snapshot
"@

git commit -m $commitMsg
Write-Host "  Commit complete" -ForegroundColor Green

# STEP 8: Tag
Write-Host "[8/8] Tagging..." -ForegroundColor Yellow

if (git tag -l "v1.0-cris-native") {
    git tag -d "v1.0-cris-native" 2>$null
}

git tag -a "v1.0-cris-native" -m "v1.0-cris-native release"
Write-Host "  Tagged: v1.0-cris-native" -ForegroundColor Green

# DONE
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  COMPLETE." -ForegroundColor Green
Write-Host ""
Write-Host "  Push to GitHub:" -ForegroundColor White
Write-Host "  git push origin main --tags" -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""