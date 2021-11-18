# MasterThesis

This is the repository of my Master thesis. It contains all the codes, files, etc.,
that were generated during the thesis. 

## Installation

To install all necessary stuff, run:

```console
$ python setup.py
```

This will setup a virtual python env and guide you through
other steps needed. Afterwards, to build the docs, run:

```console
$ ./automate.ps1 -docs
```

They will be built in the parent folder
next to the current folder. Refer to the documentation
for further instructions on how to use this repository.
Tested on Windows only. If you are running MacOS or Linux,
you might have to adjust the commands used in `automate.ps1`.
Also, `setup.py` executes some system calls that might be different
for other operating systems.

## Documentation

The code was written in such a way that it
allows for generating a documentation automatically with Sphinx.
To generate it first install Sphinx and then use:

```console
$ cd ..
$ mkdir Docs
$ cd Docs
$ sphinx-apidoc -F -H 'BatchRL' -A 'Chris' -o . '../MasterThesis/BatchRL/'
$ cp ../MasterThesis/conf.py .
$ ./make html
```

Alternatively, you can directly use the Windows Powershell script [make_doc.ps1](make_doc.ps1)
which basically runs these commands. The script also removes previously existing
folders 'Docs', so it can be used to update the documentation.

## Repo structure

This repository contains the following sub-directories.

### [Overleaf @ ...](https://github.com/chbauman/Master-ThesisOverLeaf)

This folder contains the overleaf repository
with all the latex code.

### [BatchRL](BatchRL)

This folder contains the python code
for batch reinforcement learning. The code was tested using Python version 3.6,
it is not compatible with version 3.5 or below. It was tested using PyCharm and
Visual Studio, part of the code was also run on Euler.

### [Data](Data)

Here the data is put when the code is 
executed. Should be empty on the git repo
except for the [README.md](Data/README.md).

### [Models](Models)

This folder will be used to store the
trained neural network models. 

### [Plots](Plots)

In this folder, the plots will be saved
that will be generated.

### [DocFiles](DocFiles)

This folder contains some documentation
files that will be used when creating the 
documentation with Sphinx.
