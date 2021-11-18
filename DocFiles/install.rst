============
Installation
============

To install everything needed, run:

.. code-block:: bash

    python setup.py

This will setup the virtual environment
and install all required python libraries. It will also ask 
for credentials for different accounts. This includes the access
to the NEST database and the OPCUA client. Also an email account
for sending notifications for experiment termination is desired.
These are not necessarily needed, depending on the code that is run.
But it will result in a runtime error if any of these accounts is needed.

Afterwards open Powershell in the folder
`MasterThesis` and then run:

.. code-block:: bash

    ./automate.ps1 -act

to activate the virtual environment. If you use
:option:`-docs` instead, the documentation will be 
built. Using :option:`-test` will run the unit tests
of this project.
