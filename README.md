# Barry

[Documentation online here](https://barry.readthedocs.io/en/latest/)

Modular BAO fitting code.

## Setup

1. Ensure that you have a named conda environment of at least python 3.6.
2. Clone this project onto both your local computer and a cluster computer
3. Have all dependencies installed: `pip install -r requirements.txt`
4. Update `config.yml` to include the name of your environment for activation on the HPC
5. Run any of the python files in `barry.config`.
    1. If you run on your local computer (ie `python test.py`), it will run the first MCMC run only to verify it works.
    2. If you run on a cluster (checks for cluster if the OS is centos, let me know if yours isn't), it will create a slurm job script and send out all needed runs
    3. Once all jobs have finished, copy the output from the plots folder ie `barry.config.plots.mocks` to your local computer
    4. Run the same python script and it will load in the data and create the plots.
    
Tests are included in the tests directory. Run them using pytest, `pytest -v .` in the top level directory (where this readme is).

Note that by default, we assume that the HPC system being used is slurm. If it is not, raise an issue and we'll get
something working.


## Barry Paper

Note the internal differentiation; the `configs` directory is used when performing fits and submiting jobs, whilst
the `investigations` directory is when performing investigations or tests locally.

* `configs/pk_avg.py`: Generates Figures 1 and 4
* `configs/xi_avg.py`: Generates Figures 2 and 9
* `configs/pk_individual.py`: Generates Figure 3 and 5
* `configs/ding_baoextractor.py`: Generates Figure 6
* `configs/noda_spt_vs_halofit.py`: Generates Figure 7
* `configs/noda_avg.py`: Generates Figure 8
* `configs/noda_range_lower_investigation.py`: Determines impact of shifting mink in extractor
* `configs/noda_range_upper_investigation.py`: Determines impact of shifting second k anchor in extractor
* `configs/noda_recon_covariance_investigation.py`: Determine correctness of analytic covariance matrix for Noda.
* `configs/xi_individual.py`: Generates Figure 10
* `investigations/get_consensus_measurement_individual`: Generates Figures 11 and 13
* `configs/pk_vs_xi_individual.py`: Generates Figure 12

## In-built tests

In the `tests` directory, we have three files:
* `test_datasets.py`: Will attempt to instantiate all concrete implementations of the Dataset class, ensure they have valid cosmology, and valid keys in the dictionary structure of the data. 
* `test_models.py`: Will attempt to instantiate all concrete implementations of the Model class, and then ensures that the likelihood generated at the default parameter values for the SDSS DR12 z=0.61 NGC dataset returns a finite number. Using random samples in the allowed prior range, 100 points are also randomly evaluated to ensure all return finite values.
* `test_pk2xi.py`: Validates that both the current FT and Gaussian integration methods of doing the Spherical Hankel Transform give good results.


## Adding new datasets

For examples on python codes that have digested previous datasets, look into `barry/data/sdss_dr12_pk_zbin0p61/pickle.py`.

What gets saved is a dictionary with cosmology defined inside the dataset. Pre and post-recon mocks are separated out,
and for the power spectrum data we need winfit and winpk files which define the window function, in the style as produced by
Cullan Howlett. If you want to add a new dataset but need some help, just raise an issue or send us an email.

Assuming you get the pickle made, you just need a wrapper class defining the default usage (k range, etc). See
`barry.datasets.dataset_power_spectrum.py` for examples - you can copy and paste and change the pickle name.

Also, after loading in a dataset, which will have its own smoothing scale, redshift and cosmology, you should pre-generated
the `PTGenerator` and `CambGenerator` (which are used to speed sampling up as you dont have to invoke CAMB and compute nasty
integrals in each step). To do this yourself, under `barry/cosmology` what you can do is:

1. Update `camb_generator.py` (line 199) to match your cosmology and run the file. This will generate a pickle with CAMB pregenerated.
2. Update `PT_generator.py` (line 385) to match your cosmology, update `slurm_pt_generator.job` to have the correct recon smoothing scale, and submit the job. It will generate the PT pickle.
3. Commit and push both files

This is a bit annoying, so I'm planning on writing a little file where you just give it the dataset class and it'll do this 
all for you. If that doesn't exist by the time you read this, ping me and I'll drop everything to get it done.

## Adding new models

Simply create a new class, following the examples outlined in `barry.models`.