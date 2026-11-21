$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $PythonExe)) {
    throw 'Ambiente Python ausente. Execute .\setup.ps1 primeiro.'
}
$Port = 8765
function Test-LocalPort([int]$TargetPort) {
    $Client = [System.Net.Sockets.TcpClient]::new()
    try {
        $ConnectTask = $Client.ConnectAsync('127.0.0.1', $TargetPort)
        return $ConnectTask.Wait(500) -and $Client.Connected
    } catch {
        return $false
    } finally {
        $Client.Dispose()
    }
}

$PortOpen = Test-LocalPort $Port
if (-not $PortOpen) {
    $LogDir = Join-Path $ProjectRoot 'work'
    New-Item -ItemType Directory -Force $LogDir | Out-Null
    Start-Process -FilePath $PythonExe -ArgumentList @('-m','uvicorn','app.main:app','--host','127.0.0.1','--port',"$Port") -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $LogDir 'server.log') -RedirectStandardError (Join-Path $LogDir 'server-error.log')
    for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
        Start-Sleep -Seconds 1
        $PortOpen = Test-LocalPort $Port
        if ($PortOpen) { break }
    }
}
if (-not $PortOpen) {
    throw 'A API não iniciou. Consulte work\server-error.log.'
}
Start-Process "http://127.0.0.1:$Port"
