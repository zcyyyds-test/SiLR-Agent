$ErrorActionPreference = "Stop"

$projectDir = "D:\zcy\SILR-WISE26"
$command = "cmd.exe /c D:\zcy\SILR-WISE26\scripts\run_amd_multi_action_expansion_gpu1_v2.bat"

$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $command
    CurrentDirectory = $projectDir
}

$result | Format-List *

if ($result.ReturnValue -ne 0) {
    exit 1
}
