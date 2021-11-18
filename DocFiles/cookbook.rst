========
Cookbook
========

This page explains how to use the main script, `BatchRL.py`.
You will need to `cd` into the folder `BatchRL` and activate
the virtual environment before running these commands.

Verbose mode
------------

One option that can be used in all cases, 
is :option:`-v`:
	
   python BatchRL.py -v [other_options]

In this case the output will usually be more
verbose than without that option.

Retrieve data from the NEST database
------------------------------------

If you want to loat the data from the database to your
local PC, 
simply run::

    python BatchRL.py -d --data_end_date 2020-02-21

This will retrieve, process, and store the data from 
beginning of 2019 until the specified date with the option
:option:`--data_end_date`. There is no need to specify a room
number, this will load the data for all room. Also includes
the data of the battery.

Battery
-------

Running the script with the option :option:`-b` for
battery::

    python BatchRL.py -b --data_end_date 2020-02-21

Will fit and evaluate the battery model, based on the
data that was collected up to the specified date.

Hyperparameter optimization
---------------------------

Using the option :option:`-o` for optimize will run 
the hyperparameter optimization::

   python BatchRL.py -p --room_nr 41 --hop_eval_data val

In this case you can specify the room number with the option
:option:`--room_nr`. In this case, room 41 was chosen. Refer
to the report about more information about which room this is
exactly. 
Further, you can also specify the set where the objective of
the hyperparameter tuning is evaluated, it can either be 
`val` or `test`, for validation or test set, respectively.
Not that you may also specify the data using :option:`--data_end_date`.
Using the option :option:`-in 50`, one can specify the number of
models that are fitted during the optimization. 

Evaluating models
-----------------

Using the option :option:`-m` for model evaluation, will evaluate
the the models::

   python BatchRL.py -m --train_data train_val [other_options]

It uses the corresponding hyperparameters from the hyperparameter
optimization. Therefore, the hyperparameter optimization must
be run before calling the script with this option. Additionally to 
the flags that can be used for the hyperparameter tuning, you may
specify the training set using :option:`--train_data`, possibilities 
include: "train", "train_val" and "all".

Reinforcement learning
----------------------

The reinforcement learning agent can be trained and evaluated using 
the flag :option:`-r`::

   python BatchRL.py -r -in 1000000 10000 -fl 50.0 22.0 24.0

Using the flag :option:`-in 1000000 10000`, lets you specify
the number of steps used for training and the number of steps used
for evaluation of the RL agent. :option:`-fl 50.0 22.0 24.0` lets
you specify the balance factor alpha, and the lower and the upper
temperature bounds. Note that also the previous flags, i.e. 
:option:`--data_end_date`, :option:`--room_nr` and :option:`--train_data`
may be specified to determine which model/data/room to use.

Room control using RL
---------------------

The trained reinforcement learning agent can be run on the
real system using::

   python BatchRL.py -u -fl 50.0 22.0 24.0

As in the previous case, you can specify the balance 
factor and the temperature bounds. Also the other
flags specifying the room/model/data will be needed
to determine what exactly should be controlled.

Rule-Based controller
---------------------

To run the rule-based controller, use::

   python BatchRL.py --rule_based -fl 21.0

With the flag :option:`-fl 21.0`, you specify 
that the valves will be opened, when the temperature drops
below 21.0 degrees. Note that this is only applicable for 
heating cases.

Cleanup
-------

Running the script with the option :option:`-c` for
cleanup::

    python BatchRL.py -c

Will cleanup the temporary files that were
generated during debugging and testing.

Default
-------

When not specifying any of the above options, 
the function `curr_tests()` from BatchRL.py will
be run. Add your custom code for testing or debugging 
there.

For more details about how to run the code, consider
the actual code or contact the author.
