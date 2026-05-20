param(
    [Parameter(Mandatory = $true)]
    [string]$Port,

    [string]$PythonExe = "python",

    [string]$FirmwarePath = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Resolve-Path (Join-Path $scriptDir "..")
$defaultFirmwarePath = Join-Path $projectDir "firmware_bundle\XIAO ESP32S3 Micropython\firmware.bin"
$firmwarePath = $defaultFirmwarePath
if ($FirmwarePath -ne "") {
    $firmwarePath = $FirmwarePath
    if (-not [System.IO.Path]::IsPathRooted($firmwarePath)) {
        $firmwarePath = Join-Path $projectDir $firmwarePath
    }
}

if (-not (Test-Path $firmwarePath)) {
    throw "Firmware not found at: $firmwarePath"
}

Write-Host "Using firmware: $firmwarePath"
Write-Host "Using COM port: $Port"
Write-Host ""
Write-Host "Put board in bootloader mode now:"
Write-Host "1) Hold BOOT button"
Write-Host "2) Plug USB cable"
Write-Host "3) Release BOOT"
Write-Host ""

try {
    & $PythonExe -m esptool version | Out-Null
} catch {
    Write-Host "esptool not found in this Python env, installing..."
    & $PythonExe -m pip install esptool
}

Write-Host "Erasing flash..."
& $PythonExe -m esptool --chip esp32s3 --port $Port erase-flash
if ($LASTEXITCODE -ne 0) {
    throw "erase-flash failed with exit code $LASTEXITCODE"
}

Write-Host "Writing MicroPython firmware..."
& $PythonExe -m esptool --chip esp32s3 --port $Port --baud 460800 --before default-reset --after hard-reset write-flash 0x0 $firmwarePath
if ($LASTEXITCODE -ne 0) {
    throw "write-flash failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Flash complete."
Write-Host "Next: open Thonny, connect MicroPython (ESP32), upload main.py and secrets.py."
