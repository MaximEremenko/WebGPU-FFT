param(
  [ValidateSet("tests", "bench", "all")]
  [string]$Mode = "tests",
  [int]$Port = 8011,
  [int]$TimeoutSec = 240,
  [string]$ChromePath = ""
)

$ErrorActionPreference = "Stop"

function Resolve-ChromePath {
  param([string]$Hint)
  if ($Hint -and (Test-Path $Hint)) { return $Hint }
  $candidates = @(
    "C:\Program Files\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
  )
  foreach ($p in $candidates) {
    if (Test-Path $p) { return $p }
  }
  return $null
}

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$servePy = Join-Path $here "serve.py"
if (!(Test-Path $servePy)) {
  throw "Missing serve.py at $servePy"
}

$chrome = Resolve-ChromePath -Hint $ChromePath
if (-not $chrome) {
  throw "Chrome not found. Pass -ChromePath or install Chrome."
}

Write-Host "Using Chrome: $chrome"
Write-Host "Starting local harness server on port $Port..."
$tmpOut = Join-Path $env:TEMP ("webgpufft_serve_" + [guid]::NewGuid().ToString("N") + ".out.log")
$tmpErr = Join-Path $env:TEMP ("webgpufft_serve_" + [guid]::NewGuid().ToString("N") + ".err.log")
$server = Start-Process -FilePath "python" -ArgumentList @($servePy, "$Port") -PassThru -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr

try {
  $ready = $false
  $deadline = (Get-Date).AddSeconds(30)
  while ((Get-Date) -lt $deadline) {
    if ($server.HasExited) {
      $so = if (Test-Path $tmpOut) { Get-Content -Raw $tmpOut } else { "" }
      $se = if (Test-Path $tmpErr) { Get-Content -Raw $tmpErr } else { "" }
      throw "Harness server exited early with code $($server.ExitCode).`nSTDOUT:`n$so`nSTDERR:`n$se"
    }
    try {
      $resp = Invoke-WebRequest -Uri "http://localhost:$Port/web/" -UseBasicParsing -TimeoutSec 2
      if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) {
        $ready = $true
        break
      }
    } catch {
      Start-Sleep -Milliseconds 500
    }
  }
  if (-not $ready) {
    $so = if (Test-Path $tmpOut) { Get-Content -Raw $tmpOut } else { "" }
    $se = if (Test-Path $tmpErr) { Get-Content -Raw $tmpErr } else { "" }
    throw "Harness server did not become ready on http://localhost:$Port/web/.`nSTDOUT:`n$so`nSTDERR:`n$se"
  }

  $url = "http://localhost:$Port/web/?autorun=$Mode&machine=1"
  Write-Host "Running browser autorun mode='$Mode'..."

  $args = @(
    "--headless=new",
    "--enable-unsafe-webgpu",
    "--disable-gpu-sandbox",
    "--virtual-time-budget=$($TimeoutSec * 1000)",
    "--dump-dom",
    $url
  )

  $dom = (& $chrome @args 2>&1 | Out-String)
  if ($LASTEXITCODE -ne 0) {
    throw "Chrome exited with code $LASTEXITCODE.`n$dom"
  }

  $m = [regex]::Match($dom, 'data-webgpufft-autorun="([^"]+)"')
  if (-not $m.Success) {
    $snippet = $dom.Substring(0, [Math]::Min(2000, $dom.Length))
    throw "Could not find autorun result in DOM dump. Output snippet:`n$snippet"
  }

  $json = [uri]::UnescapeDataString($m.Groups[1].Value)
  $result = $json | ConvertFrom-Json

  Write-Host ("AUTORUN {0}: ok={1} pass={2} fail={3} skip={4} total={5} ms={6}" -f `
    $result.kind, $result.ok, $result.pass, $result.fail, $result.skip, $result.total, [Math]::Round([double]$result.ms, 2))

  if (-not [bool]$result.ok -or ([int]$result.fail -gt 0)) {
    exit 1
  }
} finally {
  if ($server -and -not $server.HasExited) {
    Stop-Process -Id $server.Id -Force
  }
  if (Test-Path $tmpOut) { Remove-Item $tmpOut -Force -ErrorAction SilentlyContinue }
  if (Test-Path $tmpErr) { Remove-Item $tmpErr -Force -ErrorAction SilentlyContinue }
}
