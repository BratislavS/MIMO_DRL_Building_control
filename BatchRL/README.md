# All the python code 

The file `BatchRL.py` is the main function, run this to run
everything. 

## Running tests

The script `run_tests.ps1` let's you run some unit tests.
If you are not using Powershell, just type:
```console
$ python -m unittest discover .
```
and it should run. It does not contain all tests, there is 
another function in the main script `BatchRL.py` that runs some 
tests that need some more time to run.

## Running on Euler

### Connecting to Euler using Putty

First set uf a VPN connection if not in ETH network.
Then run putty, select `euler.ethz.ch` as Host Name and connect,
it will ask for credentials and then you should be logged in.

### Connecting to Euler from Linux terminal

Use:
```console
$ ssh username@euler.ethz.ch
```

### Setting things up

You can use git to clone this repository, then you
do not have to copy the code manually.
You might need to install some additional
python libraries, do this using the flag `--user`.

### Running on Euler

To load the necessary libraries, run:
```console
$ module load python
$ module load hdf5
```
or, alternatively, execute the script that does that:
```console
$ source ../load_euler
```
Also remember to copy the data if it has changed 
since the last time. The command for doing this
is in [Data](../Data). You also might need to make the 
script files executable, for that purpose, use: `chmod +x script.ext`.
