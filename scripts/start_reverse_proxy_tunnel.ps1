param(
    [string]$HostAlias = "humanoid-server",
    [int]$LocalPort = 7890,
    [int]$RemotePort = 7890,
    [string]$PidFile = "$env:TEMP\humanoid_agent_reverse_proxy_tunnel.pid"
)

$localListener = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalAddress -eq "127.0.0.1" -and $_.LocalPort -eq $LocalPort } |
    Select-Object -First 1

if (-not $localListener) {
    $tcpProbe = Test-NetConnection -ComputerName 127.0.0.1 -Port $LocalPort -WarningAction SilentlyContinue
}

if (-not $localListener -and -not $tcpProbe.TcpTestSucceeded) {
    throw "No local proxy is listening on 127.0.0.1:$LocalPort. Start Clash first or pass -LocalPort."
}

if (Test-Path $PidFile) {
    $existingPid = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($existingPid) {
        $existing = Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue
        if ($existing) {
            Write-Output "Tunnel already running. PID=$existingPid"
            exit 0
        }
    }
}

$args = @(
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=6",
    "-N",
    "-R", "127.0.0.1:${RemotePort}:127.0.0.1:${LocalPort}",
    $HostAlias
)

$proc = Start-Process -FilePath "ssh" -ArgumentList $args -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 2

if ($proc.HasExited) {
    throw "SSH reverse proxy tunnel exited immediately with code $($proc.ExitCode)."
}

Set-Content -Path $PidFile -Value $proc.Id
Write-Output "Tunnel started. PID=$($proc.Id), server endpoint=http://127.0.0.1:$RemotePort"
