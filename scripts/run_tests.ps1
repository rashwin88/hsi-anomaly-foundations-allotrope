# Run the app/ test suite inside the worker container.
# tests/ is not baked into the image, so it is bind-mounted here at run time.
# Usage: scripts\run_tests.ps1 [extra pytest args]
# Keep this file pure ASCII - PowerShell 5.1 reads .ps1 as ANSI.
param([Parameter(ValueFromRemainingArguments = $true)] $PytestArgs)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
if (-not (Test-Path "$root\docker\.env")) {
    throw "docker/.env is missing - copy docker/.env.example and fill it in."
}
if (-not $PytestArgs) { $PytestArgs = @("-q") }
$marks = "not large_files and not large_benchmarks and not network_access"
$dockerArgs = @(
    "compose", "-f", "$root\docker\docker-compose.yml", "run", "--rm", "--no-deps",
    "-v", "$root\tests:/srv/tests",
    "-v", "$root\pytest.ini:/srv/pytest.ini",
    "worker", "python", "-m", "pytest", "tests", "-m", $marks
) + $PytestArgs
docker @dockerArgs
