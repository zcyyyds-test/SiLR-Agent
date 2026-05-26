$ErrorActionPreference = "Stop"

$projectDir = "D:\zcy\SILR-WISE26"
$command = "cmd.exe /c D:\zcy\SILR-WISE26\scripts\launch_qwen3_14b_gpu1_8004_wsl.bat"

$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $command
    CurrentDirectory = $projectDir
}

$result | Format-List *

if ($result.ReturnValue -ne 0) {
    exit 1
}
