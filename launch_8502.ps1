# launch_8502.ps1
# Run the Tri-HB integrated Streamlit app on port 8502.
# Usage:   .\launch_8502.ps1
# (If PowerShell blocks the script, run once:
#  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)

$ErrorActionPreference = "Stop"

# Run from this script's folder so the app finds Tri-HB.py and wave_damage.py
Set-Location -Path $PSScriptRoot

# Pick the Python launcher: prefer 'py -3.13', fall back to 'python'
$python = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $python = @("py", "-3.13")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python = @("python")
} else {
    Write-Error "No Python interpreter found (need 'py' or 'python' on PATH)."
}

Write-Host "Using Python:" ($python -join " ")
& $python[0] $python[1..($python.Length - 1)] -m streamlit run tri_hb_integrated.py --server.port 8502
