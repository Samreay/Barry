# Setting up Barry on NERSC

1. Git clone the Barry repository onto both your local computer and NERSC: `git clone https://github.com/Samreay/Barry.git`
2. Load Arnaud de Mattia's cosmodesi environment
    ```
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    ```
3. Install the additional python requirements `cd Barry && pip install -r requirements.txt`
4. Add `export HPC=nersc` to your `~/.bashrc.ext` file and `source ~/.bashrc.ext` it.
5. Barry should now be more or less good to go. Now let's run some tests. From the main Barry directory try:
    1. `pytest -v`: This will do some unit tests. They should all pass.
    2. `cd config/examples && python test_emcee_mock_avg.py`. This should submit a fit to the BOSS DR12 z3 NGC mock
    average and return a chain. Will take a few minutes to run. The auto generated job scripts and 
    SLURM output files can be found in `job_files/` and `out_files/` respectively. Once the chain has finished, 
    you can find it in `plots/test_emcee_mock_avg/output`. 
    3. Now run `python test_emcee_mock_avg.py plot`, or copy the directory structure in the previous step (from `plots/` downwards) 
    into `config/examples/` on your local computer and run `python test_emcee_mock_avg.py` again. This will now 
    analyse the chain and put some plots in `plots/test_emcee_mock_avg/`.
6. If you are feeling adventurous you can try looking at and running some of the other `test_` codes in `examples`.