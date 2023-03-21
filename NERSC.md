# Setting up Barry on NERSC

1. Git clone the Barry repository onto both your local computer and NERSC: `git clone https://github.com/Samreay/Barry.git`
2. Load Arnaud de Mattia's cosmodesi environment
    ```
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    ```
3. Add `export HPC=perlmutter` (or `export HPC=cori` if using cori) to your `~/.bashrc.ext` file and `source ~/.bashrc.ext` it. 
4. Barry should now be more or less good to go.
