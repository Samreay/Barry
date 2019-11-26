.. Barry

Barry: Easy BAO model fitting!
=================================

Welcome to the online documentation for Barry.

Check out the `config file for examples on how to use Barry <https://github.com/Samreay/Barry/tree/master/config>`_.

Or read through the


Installation
------------

Barry requires the following dependencies::

    numpy
    scipy
    matplotlib
    chainconsumer
    emcee
    dynesty
    pandas
    camb
    hankel
    pyyaml


You can install Barry either locally, or if you want
to :code:`pip` install it, run::

    pip install barry

Authors
-------

* Samuel Hinton: samuelreay@gmail.com
* Cullan Howlett: c.howlett@uq.edu.au

Issues
------
Please raise issues, bugs and feature requests `on the Github Issues Page <https://github.com/Samreay/Barry/issues>`_.

How to Use
----------

1. Ensure that you have a named conda environment of at least python 3.6.
2. Clone this project onto both your local computer and a cluster computer
3. Have all dependencies installed: :code:`pip install -r requirements.txt`
4. Update :code:`config.yml` to include the name of your environment for activation on the HPC
5. Run any of the python files in :code:`barry.config`.
    1. If you run on your local computer (ie :code:`python test.py`), it will run the first MCMC run only to verify it works.
    2. If you run on a cluster (checks for cluster if the OS is centos, let me know if yours isn't), it will create a slurm job script and send out all needed runs
    3. Once all jobs have finished, copy the output from the plots folder ie :code:`barry.config.plots.mocks` to your local computer
    4. Run the same python script and it will load in the data and create the plots.

Check out the API below or browse the examples used to make the Barry paper. For any questions, flick me an email (samuelreay@gmail.com).

Index
-----

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   barry_api