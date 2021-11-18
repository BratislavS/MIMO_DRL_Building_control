<#
.SYNOPSIS
    Script to run evaluation of a few models.

.DESCRIPTION
    Runs the python script BatchRL.py with the -r option
    to run the room model using the trained
    RL agent. Does this using different parameters, i.e. with and without
    the battery, (not) using the physical model.

.NOTES
    Author: Christian Baumann
    Last Edit: 2019-01-14
    Version 1.0 - initial release
#>

# Array definition
$true_false = @("t","f")
$pen_facs = @("2.5", "50.0")

# Performance evaluations
foreach ($add_bat in $true_false){
   foreach ($p in $pen_facs){
      foreach ($phys in $true_false){
         # Exclude case of battery model where not the
         # physical model was used.
         if (($add_bat -eq "f") -or ($phys -eq "t")){
            python .\BatchRL.py -r -v -fl $p -bo $add_bat t $phys
         }
      }
   }
}