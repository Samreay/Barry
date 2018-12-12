# Barry

Modular BAO fitting code.

Requirements to run fits:

1. Have all dependencies installed: `pip install -r requirements.txt`
2. Ensure that you have a named conda environment of at least python 3.6.
3. Clone this project onto both your local computer and a cluster computer
4. Update `barry.framework.config.py` to include the name of your environment
5. Run any of the python files in `barry.config`.
    1. If you run on your local computer (ie `python test.py`), it will run the first MCMC run only to verify it works.
    2. If you run on a cluster (checks for cluster if the OS is centos, let me know if yours isn't), it will create a slurm job script and send out all needed runs
    3. Once all jobs have finished, copy the output from the plots folder ie `barry.config.plots.mocks` to your local computer
    4. Run the same python script and it will load in the data and create the plots.