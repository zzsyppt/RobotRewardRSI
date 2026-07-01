param(
    [string]$PidFile = "$env:TEMP\humanoid_agent_reverse_proxy_tunnel.pid"
)

$stopped = $false

if (Test-Path $PidFile) {
    $existingPid = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($existingPid) {
        $existing = Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue
        if ($existing) {
            Stop-Process -Id $existing.Id -Force
            Write-Output "Stopped tunnel PID=$existingPid"
            $stopped = $true
        }
    }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

if (-not $stopped) {
    Write-Output "No tunnel PID from $PidFile was running."
}
