<#
.SYNOPSIS
    Script with some example commands.

.DESCRIPTION
    Gives an overview of some PS commands that are
    used frequently. Look at the source code, the actual output
    is pretty meaningless on its own.

.PARAMETER temp
    A dummy parameter.

.EXAMPLE
    It can e.g. be run as:
    PS C:\> ./ps_test -temp 5

.NOTES
    Author: Christian Baumann
    Last Edit: 2019-01-14
    Version 1.0 - initial release
#>

# Parameter definition
param(
[int]$temp,
[switch]$b = $false,
[string]$s = "hoi")

Write-Host "Got parameter: $temp, b: $b"

# Variable definition
$v = 15
Write-Host "This is a variable: $v"

# Array definition
$array = @("val_1","val_2","val_3")

# For loop over array
for ($i=0; $i -lt $array.length; $i++){
    $ai = $array[$i]
    Write-Host "Array element at position $i is: $ai"
    # Alternatively: echo $array[$i]
}

# Or with for each
foreach ($i in $array){
   Write-Host $i

   # If clause
   if ($i -eq "val_1") {
       Write-Host hello
   }
}

# Functions

# Definition
Function Hello ($name)
{
    Write-Host "hoi $name"
}

# Call the function
Hello ("Hans")

# Activate python environment
Invoke-Expression "$($PSScriptRoot)/venv/Scripts/Activate.ps1"
