# Simple dependency installer for 261-dacn.
# Usage:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\install_deps.ps1

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Requirements = Join-Path $Root "requirements.txt"
$SumoWingetId = "EclipseFoundation.SUMO"
$PythonWingetId = "Python.Python.3.12"

function Ask-YesNo([string]$Question) {
    while ($true) {
        $answer = Read-Host "$Question [Y/n]"
        if ([string]::IsNullOrWhiteSpace($answer) -or $answer -match '^(y|yes)$') { return $true }
        if ($answer -match '^(n|no)$') { return $false }
        Write-Host "Please answer Y or N."
    }
}

function Test-Cmd([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-RequirementPackages {
    if (-not (Test-Path $Requirements)) { return @() }
    return Get-Content $Requirements |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -and -not $_.StartsWith("#") } |
        ForEach-Object {
            # strip extras / version pins: "pandas>=2.0" -> "pandas"
            if ($_ -match '^\s*([A-Za-z0-9_.-]+)') { $Matches[1] }
        }
}

function Test-PipPackage([string]$Name) {
    python -m pip show $Name 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Find-SumoHome {
    $candidates = @()
    if ($env:SUMO_HOME) { $candidates += $env:SUMO_HOME }
    $candidates += @(
        "C:\Program Files\Eclipse\Sumo",
        "C:\Program Files (x86)\Eclipse\Sumo",
        "$env:LOCALAPPDATA\Programs\Eclipse\Sumo"
    )
    foreach ($root in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if (-not $root) { continue }
        $eclipse = Join-Path $root "Eclipse"
        if (Test-Path $eclipse) {
            Get-ChildItem $eclipse -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match "Sumo" } |
                ForEach-Object { $candidates += $_.FullName }
        }
    }
    foreach ($path in ($candidates | Select-Object -Unique)) {
        if (Test-Path (Join-Path $path "bin\sumo-gui.exe")) { return $path }
    }
    return $null
}

function Set-SumoEnv([string]$SumoHome) {
    [Environment]::SetEnvironmentVariable("SUMO_HOME", $SumoHome, "User")
    $env:SUMO_HOME = $SumoHome

    $bin = Join-Path $SumoHome "bin"
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $userPath) { $userPath = "" }
    $parts = @($userPath -split ";" | Where-Object { $_ })
    if ($parts -notcontains $bin) {
        [Environment]::SetEnvironmentVariable("Path", (($parts + $bin) -join ";"), "User")
        Write-Host "Added to user PATH: $bin"
    }
    $env:Path = "$bin;$env:Path"
    Write-Host "SUMO_HOME=$SumoHome"
}

Write-Host "261-dacn dependency check" -ForegroundColor Green

# --------------------------------------------------
# 1) Python + pip + requirements.txt
# --------------------------------------------------
Write-Host ""
Write-Host "==> 1) Python / pip / requirements.txt" -ForegroundColor Cyan

if (-not (Test-Cmd "python")) {
    Write-Host "Python is not installed (or not on PATH)."
    if (Ask-YesNo "Install Python 3.12 with winget now?") {
        if (-not (Test-Cmd "winget")) {
            Write-Host "winget not found. Install Python from https://www.python.org/downloads/ then re-run." -ForegroundColor Yellow
            exit 1
        }
        winget install -e --id $PythonWingetId --accept-package-agreements --accept-source-agreements
        Write-Host "Python installed. Re-open this terminal, then run .\install_deps.ps1 again." -ForegroundColor Yellow
        exit 0
    }
    Write-Host "Skipped. Install Python manually, then re-run."
    exit 1
}

Write-Host "Python: $(python --version 2>&1)"

python -m pip --version 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip is not available for this Python."
    if (Ask-YesNo "Bootstrap pip with 'python -m ensurepip --upgrade' now?") {
        python -m ensurepip --upgrade
        python -m pip install --upgrade pip
    } else {
        Write-Host "Skipped. Install pip, then re-run."
        exit 1
    }
} else {
    Write-Host "pip: $(python -m pip --version 2>&1)"
}

if (-not (Test-Path $Requirements)) {
    Write-Host "requirements.txt not found — skipping pip packages."
} else {
    $packages = @(Get-RequirementPackages)
    if ($packages.Count -eq 0) {
        Write-Host "requirements.txt has no packages (comments only) — nothing to install with pip."
    } else {
        $missing = @()
        foreach ($pkg in $packages) {
            if (Test-PipPackage $pkg) {
                Write-Host "OK  $pkg"
            } else {
                Write-Host "MISS $pkg" -ForegroundColor Yellow
                $missing += $pkg
            }
        }

        if ($missing.Count -gt 0) {
            Write-Host "Missing pip packages: $($missing -join ', ')"
            if (Ask-YesNo "Run 'pip install -r requirements.txt' now?") {
                python -m pip install -r $Requirements
            } else {
                Write-Host "Skipped pip install. You can run later:"
                Write-Host "  python -m pip install -r requirements.txt"
            }
        } else {
            Write-Host "All requirements.txt packages are installed."
        }
    }
}

# --------------------------------------------------
# 2) winget
# --------------------------------------------------
Write-Host ""
Write-Host "==> 2) winget" -ForegroundColor Cyan

$hasWinget = Test-Cmd "winget"
if ($hasWinget) {
    Write-Host "winget: OK"
} else {
    Write-Host "winget not found." -ForegroundColor Yellow
    Write-Host "Install 'App Installer' from Microsoft Store, or install SUMO manually from https://eclipse.dev/sumo/"
}

# --------------------------------------------------
# 3) SUMO
# --------------------------------------------------
Write-Host ""
Write-Host "==> 3) Eclipse SUMO" -ForegroundColor Cyan

$sumoHome = Find-SumoHome
if ($sumoHome) {
    Write-Host "SUMO found: $sumoHome"
    Set-SumoEnv $sumoHome
} else {
    Write-Host "SUMO (sumo-gui) not found."
    if (-not $hasWinget) {
        Write-Host "Cannot auto-install without winget. Install SUMO manually, set SUMO_HOME, then re-run."
        exit 1
    }
    if (Ask-YesNo "Install Eclipse SUMO with winget now?") {
        winget install -e --id $SumoWingetId --accept-package-agreements --accept-source-agreements
        Start-Sleep -Seconds 2
        $sumoHome = Find-SumoHome
        if (-not $sumoHome) {
            Write-Host "SUMO installed, but path not detected yet. Re-open the terminal and re-run .\install_deps.ps1" -ForegroundColor Yellow
            exit 0
        }
        Set-SumoEnv $sumoHome
    } else {
        Write-Host "Skipped. Install later with:"
        Write-Host "  winget install -e --id $SumoWingetId"
        exit 1
    }
}

# --------------------------------------------------
# Done
# --------------------------------------------------
Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "Next:"
Write-Host "  python sim_sumo.py -mode light"
Write-Host "If PATH/SUMO_HOME look wrong, open a new terminal first."
