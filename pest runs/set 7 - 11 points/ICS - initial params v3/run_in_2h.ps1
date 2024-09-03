# Get the directory of the script
$scriptDirectory = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent

# Define the commands to be executed
$commands = {
    # Change to the script directory
    Set-Location -Path $using:scriptDirectory

    # Execute the script 
    Start-Process "C:\pest17\pest.exe eeek"
}

# Get the current date and time
$startTime = Get-Date

# Calculate the execution time (2 hours from now)
$executionTime = $startTime.AddHours(2)

# Display the scheduled time
Write-Output "Commands will execute at $executionTime in $scriptDirectory"

# Wait until the scheduled time
while ((Get-Date) -lt $executionTime) {
    Start-Sleep -Seconds 60
}

# Execute the commands
& $commands
