$pidPath = Join-Path $PSScriptRoot "dashboard.pid"
if (-not (Test-Path $pidPath)) {
    Write-Host "dashboard.pid not found."
    exit 0
}

$targetPid = Get-Content $pidPath | Select-Object -First 1
if (-not $targetPid) {
    Write-Host "dashboard.pid is empty."
    exit 0
}

$process = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
if ($process) {
    Stop-Process -Id $targetPid
    Write-Host "Stopped dashboard process $targetPid."
} else {
    Write-Host "Dashboard process $targetPid is not running."
}
