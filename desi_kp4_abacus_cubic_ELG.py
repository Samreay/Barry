import os
import sys
import pickle
import numpy as np

sys.path.append("../Barry/")     # Change this so that it points to where you have Barry installed

from barry.samplers import DynestySampler
from barry.config import setup
from barry.models import PowerBeutler2017, CorrBeutler2017 # Beutler et al 2017 methods for calculation power spectrum, correlation function 
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESI_KP4 # classes with methods for fitting data? 
from barry.datasets.dataset_correlation_function import CorrelationFunction_DESI_KP4
from barry.fitter import Fitter # manages model fitting 
from barry.models.model import Correction # class for applying corrections to the likelihood function 

if __name__ == "__main__":
    
    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

    # Set up the Fitting class and Dynesty sampler with 500 live points. 
    # Set remove_output=False to make sure that we don't delete/overwrite existing chains in the same directory.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = DynestySampler(temp_dir=dir_name, nlive=500)
    
    # The optimal sigma values we found when fitting the mocks with fixed alpha/epsilon
    sigma_nl_par = {None: 8.7, "sym": 5.4}
    sigma_nl_perp = {None: 4.0, "sym": 1.5}
    sigma_s = {None: 3.5, "sym": 0.0}
    
    # Loop over the mocktypes
    allnames = []
    mocknames = ['desi_kp4_abacus_cubicbox', 'desi_kp4_abacus_cubicbox_cv']
    for i, mockname in enumerate(mocknames):

        # Loop over pre- and post-recon measurements
        for recon in [None, "sym"]:

            # Create the data. We'll fit monopole, quadrupole between k=0.02 and 0.3.
            # First load up mock mean and add it to the fitting list.
            dataset_pk = PowerSpectrum_DESI_KP4(
                recon=recon,
                fit_poles=[0, 2],
                min_k=0.02,
                max_k=0.30,
                realisation=None,          # realisation=None while load the average of all the realisations
                num_mocks=1000,            # Used for Hartlap/Sellentin correction if correction=Correction.HARTLAP or Correction.SELLENTIN
                reduce_cov_factor=1,       # Use standard covariance, even for the average
                datafile=mockname+"_pk_elg.pkl",
                #data_location="../prepare_data/",
                data_location="cosmodesi_KP4ELG_examplecode_make_picklefiles/",
            )

            dataset_xi = CorrelationFunction_DESI_KP4(
                recon=recon,
                fit_poles=[0, 2],
                min_dist=52.0,
                max_dist=150.0,
                realisation=None,
                num_mocks=1000,
                reduce_cov_factor=1,
                datafile=mockname+"_xi_elg.pkl",
                #data_location="../prepare_data/",
                data_location="cosmodesi_KP4ELG_examplecode_make_picklefiles/",
            )

            # Set up the appropriate model for the power spectrum
            model = PowerBeutler2017(
                recon=dataset_pk.recon,                   
                isotropic=dataset_pk.isotropic,
                marg="full",                              # Analytic marginalisation
                poly_poles=dataset_pk.fit_poles,
                correction=Correction.NONE,               # No covariance matrix debiasing
                n_poly=6,                                 # 6 polynomial terms for P(k)
            )
            
            # Set Gaussian priors for the BAO damping centred on the optimal values 
            # found from fitting with fixed alpha/epsilon and with width 2 Mpc/h
            model.set_default("sigma_nl_par", sigma_nl_par[recon], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
            model.set_default("sigma_nl_perp", sigma_nl_perp[recon], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
            model.set_default("sigma_s", sigma_s[recon], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

            # Load in the proper DESI BAO template rather than Barry computing its own.
            # pktemplate = np.loadtxt("../prepare_data/DESI_Pk_template.dat")
            pktemplate = np.loadtxt("cosmodesi_KP4ELG_examplecode_make_picklefiles/DESI_Pk_template.dat")
            
            model.kvals, model.pksmooth, model.pkratio = pktemplate.T

            # Give the data+model pair a name and assign it to the list of fits
            name = dataset_pk.name + " mock mean"
            fitter.add_model_and_dataset(model, dataset_pk, name=name)
            allnames.append(name)
            
            # Now add the individual realisations to the list
            for j in range(len(dataset_pk.mock_data)):
                dataset_pk.set_realisation(j)
                name = dataset_pk.name + f" realisation {j}"
                fitter.add_model_and_dataset(model, dataset_pk, name=name)
                allnames.append(name)

            model = CorrBeutler2017(
                recon=dataset_xi.recon,
                isotropic=dataset_xi.isotropic,
                marg="full",
                poly_poles=dataset_xi.fit_poles,
                correction=Correction.NONE,
                n_poly=4,    # 4 polynomial terms for Xi(s)
            )

            # Set Gaussian priors for the BAO damping centred on the optimal values 
            # found from fitting with fixed alpha/epsilon and with width 2 Mpc/h
            model.set_default("sigma_nl_par", sigma_nl_par[recon], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
            model.set_default("sigma_nl_perp", sigma_nl_perp[recon], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
            model.set_default("sigma_s", sigma_s[recon], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

            #pktemplate = np.loadtxt("../prepare_data/DESI_Pk_template.dat")
            pktemplate = np.loadtxt("cosmodesi_KP4ELG_examplecode_make_picklefiles/DESI_Pk_template.dat")
            model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

            name = dataset_xi.name + " mock mean"
            fitter.add_model_and_dataset(model, dataset_xi, name=name)
            allnames.append(name)

            # Now add the individual realisations to the list
            for j in range(len(dataset_xi.mock_data)):
                dataset_xi.set_realisation(j)
                name = dataset_xi.name + f" realisation {j}"
                fitter.add_model_and_dataset(model, dataset_xi, name=name)
                allnames.append(name)

    print(allnames)
                
    # Set the sampler (dynesty) and assign 1 walker (processor) to each. If we assign more than one walker, for dynesty
    # this means running independent chains which will then get added together when they are loaded in.
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)
    
    # If this is being run for the first time (i.e., not via a submission script), dump the entire fitter class to a file
    # so we can use it to read in the chains/models/datasets for plotting in a jupyter notebook
    if len(sys.argv) == 1:
        outfile = fitter.temp_dir+pfn.split("/")[-1]+".fitter.pkl"
        with open(outfile, 'wb') as pickle_file:
            pickle.dump(fitter, pickle_file)