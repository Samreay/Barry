# Setting up Barry on NERSC

1. Git clone the Barry repository onto both your local computer and NERSC: `git clone https://github.com/Samreay/Barry.git`
2. Load Arnaud de Mattia's cosmodesi environment
    ```
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    ```
3. Install the additional python requirements `cd Barry && pip install -r requirements.txt`
4. Add `export HPC=nersc` to your `~/.bashrc.ext` file
7. Barry should now be more or less good to go. Now let's run some tests. From the main Barry directory try:
    1. `cd barry && python generate.py`: This will run through all the datasets and models and check
    that you have the corresponding CAMB power spectrum templates and non-linear integrals. Unless you have
    added a new dataset or model, they should all be present. If any are missing, Barry will create a job script 
    to compute it in `jobs/` and submit it. Give it some time to run.
    2. `cd ../ && pytest -v`: This will do some unit tests. They should all pass, but might not if `generate.py`
    has already identified that some templates are missing.
    3. `cd config/examples && python test_emcee_mock_avg.py`. This should submit a fit to the BOSS DR12 z3 NGC mock
    average and return a chain. Will only take a couple of minutes to run. The auto generated job scripts and 
    SLURM output files can be found in `job_files/` and `out_files` respectively. Once the chain has finished, 
    you can find it in `plots/test_emcee_mock_avg/output`. Copy this directory structure (from `plots/` downwards) 
    into `config/examples/` on your local laptop and run `python test_emcee_mock_avg.py` again. This will now 
    analyse the chain and put some plots in `plots/test_emcee_mock_avg/`.
    4. If you are feeling adventurous you can try looking at and running some of the other `test_` codes in `examples`.