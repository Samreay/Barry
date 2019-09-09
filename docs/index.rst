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

1. Check out / download / clone / fork Barry to your local machine.
2. Customise the :code:`config.yml` file for your usage.
3. Implement some config file which loads models and datasets into the fitter. Check the entire :code:`config` directory for examples.
4. Copy / re-clone your version of Barry to your HPC system. Output will go in the :code:`plots/yourfile` directory.
5. Start a fitting run by invoking your configuration file: :code:`python yourfile.py`
6. Once all jobs have finished, either copy the "plots/yourfile" directory back to your local machine and then run :code:`python yourfile.py` again, or run :code:`python yourfile.py -1` and get the output plots.

Check out the API below or browse the examples used to make the Barry paper. For any questions, flick me an email (samuelreay@gmail.com).

Index
-----

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   barry_api