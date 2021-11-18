<#
.SYNOPSIS
    Script that runs various commands commands.

.DESCRIPTION
    What is run depends on the flags that are passed. Pass '-act'
    to activate the python environment, '-docs' to build
    the documentations or one of '-cp_plots', '-cp_hop', '-cp_data'
    to copy plot / hyperopf / data from / to Euler via scp. (Copying
    data needs a VPN connection to ETH.)

.PARAMETER cp_plots
    Set if you want to copy plots from Euler.
    You will need a vpn connection!
.PARAMETER act
    Set if you want to activate the python environment.
.PARAMETER docs
    Set if you want to build the documentation.
.PARAMETER test
    Run the unit tests. Activates the python env if it has not
    yet been done.

.EXAMPLE
    If you want to do everything, run:
    PS C:\> ./automate -cp_plots -act -docs

.NOTES
    Author: Christian Baumann
    Last Edit: 2019-01-15
    Version 1.0 - initial release
#>

# Parameter definition
param(
[switch]$cp_plots = $false,
[switch]$cp_data = $false,
[switch]$cp_hop = $false,
[switch]$cp_rl = $false,
[switch]$act = $false,
[switch]$test = $false,
[switch]$docs = $false)

# Copy plots from Euler
if ($cp_plots){
    scp -rp chbauman@euler.ethz.ch:MT/MasterThesis/Plots/ ./EulerPlots/
}
# Copy data to Euler
if ($cp_data){
    Invoke-Expression "scp -rp $($PSScriptRoot)/Data/Datasets/ chbauman@euler.ethz.ch:MT/MasterThesis/Data/"
}
# Copy hyperoptimization data from Euler
if ($cp_hop){
    Invoke-Expression "scp -rp chbauman@euler.ethz.ch:MT/MasterThesis/Models/Hop/ $($PSScriptRoot)/Models/"
}
# Copy RL agents from Euler
if ($cp_rl){
    Invoke-Expression "scp -rp chbauman@euler.ethz.ch:MT/MasterThesis/Models/RL/ $($PSScriptRoot)/Models"
}

# Run tests
if ($test){
    Invoke-Expression "$($PSScriptRoot)/venv/Scripts/Activate.ps1"
    Invoke-Expression "cd $($PSScriptRoot)/BatchRL; ./run_tests.ps1"
}

# Activate python env
if ($act){
    Invoke-Expression "$($PSScriptRoot)/venv/Scripts/Activate.ps1"
}

# Build the docs
if ($docs){
    Invoke-Expression "$($PSScriptRoot)/venv/Scripts/Activate.ps1"
    Invoke-Expression "cd $PSScriptRoot; ./make_doc.ps1"
}