from functools import lru_cache

import numpy as np
import inspect
import os
import logging


# TODO: Add options for mnu, h0 default, omega_b, etc


@lru_cache(maxsize=32)
def getCambGenerator(
    redshift=0.51, om_resolution=101, h0_resolution=1, h0=0.676, ob=0.04814, ns=0.97, mnu=0.0, recon_smoothing_scale=21.21, neff=3.044,
    neff_resolution=1, vary_neff=False,):
    
    return CambGenerator(
        redshift=redshift,
        om_resolution=om_resolution,
        h0_resolution=h0_resolution,
        h0=h0,
        ob=ob,
        ns=ns,
        mnu=mnu,
        recon_smoothing_scale=recon_smoothing_scale,
        vary_neff=vary_neff, 
        neff=neff,
        neff_resolution=neff_resolution,
    )


def Omega_m_z(omega_m, z):
    """
    Computes the matter density at redshift based on the present day value.

    Assumes Flat LCDM cosmology, which is fine given this is also assumed in CambGenerator. Possible improvement
    could be to tabulate this using the CambGenerator so that it would be self consistent for non-LCDM cosmologies.

    :param omega_m: the matter density at the present day
    :param z: the redshift we want the matter density at
    :return: the matter density at redshift z
    """
    return omega_m * (1.0 + z) ** 3 / E_z(omega_m, z) ** 2


def E_z(omega_m, z):
    """
    Compute the E-function; the ratio of the Hubble parameter at redshift z to the Hubble-Lemaitre constant.

    Assumes Flat LCDM cosmology, which is fine given this is also assumed in CambGenerator. Would not be necessary if
    we tabulated Omega_m_z using the CambGenerator.

    :param omega_m: the matter density at the present day
    :param z: the redshift we want the E-function at
    :return: The E-function at redshift z given the matter density
    """
    return np.sqrt((1.0 + z) ** 3 * omega_m + (1.0 - omega_m))


class CambGenerator(object):
    """An object to generate power spectra using camb and save them to file.

    Useful because computing them in a likelihood step is insanely slow.
    """

    def __init__( 
        self, redshift=0.61, om_resolution=101, h0_resolution=1, h0=0.676, ob=0.04814, ns=0.97, mnu=0.0, recon_smoothing_scale=21.21, vary_neff=False, neff=3.044, neff_resolution=1,
    ):
        """
        Precomputes CAMB for efficiency. Access ks via self.ks, and use get_data for an array
        of both the linear and non-linear power spectrum
        """
        self.logger = logging.getLogger("barry")
        self.om_resolution = om_resolution
        self.h0_resolution = h0_resolution
        self.vary_neff = vary_neff
        self.neff_resolution=neff_resolution
        self.h0 = h0
        self.redshift = redshift
        checkfor_singleval_neff = (not vary_neff or neff_resolution == 1)
        self.singleval = True if om_resolution == 1 and h0_resolution == 1 and checkfor_singleval_neff else False
        self.data_dir = os.path.normpath(os.path.dirname(inspect.stack()[0][1]) + "/../generated/")
        hh = int(h0 * 10000)
        self.filename_unique = f"{int(self.redshift * 1000)}_{self.om_resolution}_{self.h0_resolution}_{hh}_{int(ob * 10000)}_{int(ns * 1000)}_{int(mnu * 10000)}"
        if vary_neff: 
            self.filename_unique = f"{int(self.redshift * 1000)}_{self.om_resolution}_{self.h0_resolution}_{self.neff_resolution}_{hh}_{int(ob * 10000)}_{int(ns * 1000)}_{int(mnu * 10000)}"
        self.filename = self.data_dir + f"/camb_{self.filename_unique}.npy"
        self.k_min = 1e-5
        self.k_max = 100
        self.k_num = 2000
        self.ks = np.logspace(np.log(self.k_min), np.log(self.k_max), self.k_num, base=np.e)
        self.recon_smoothing_scale = recon_smoothing_scale
        self.smoothing_kernel = np.exp(-self.ks**2 * self.recon_smoothing_scale**2 / 2.0)

        self.omch2s = np.linspace(0.05, 0.3, self.om_resolution)
        self.omega_b = ob
        self.ns = ns
        self.mnu = mnu
        self.neff = neff
        
        if h0_resolution == 1:
            self.h0s = [h0]
        else:
            self.h0s = np.linspace(0.6, 0.8, self.h0_resolution)
            
        if neff_resolution == 1:
            self.neffs = [neff]
        else:
            self.neffs = np.linspace(0., 5., self.neff_resolution)

        self.data = None
        if not vary_neff:
            self.logger.info(f"Creating CAMB data with {self.om_resolution} x {self.h0_resolution}")
        else:
            self.logger.info(f"Creating CAMB data with {self.om_resolution} x {self.h0_resolution} x {self.neff_resolution}")

    def load_data(self, can_generate=False):
        if not os.path.exists(self.filename):
            print(self.filename)
            if not can_generate:
                msg = "Data does not exist and this isn't the time to generate it!"
                self.logger.error(msg)
                raise ValueError(msg)
            else:
                self.data = self._generate_data()
        else:
            self.data = np.load(self.filename)
            self.logger.info("Loading existing CAMB data")

    @lru_cache(maxsize=512)
    def get_data(self, om=0.31, h0=None, neff=None): # gets the data at given cosmo, loads it if its already computed but calculates it if needed also. 
        """Returns the sound horizon, the linear power spectrum, and the halofit power spectrum at self.redshift"""
        if h0 is None:
            h0 = self.h0
        if neff is None:
            neff = self.neff
        if self.data is None:
            # If we are not interested in varying om, we can run CAMB this once to avoid precomputing
            if self.singleval:
                self.logger.info(f"Running CAMB")
                self.data = self._generate_data(savedata=False)[0, 0]
            else:
                self.load_data()
        if self.singleval:
            data = self.data
        else:
            omch2 = (om - self.omega_b) * h0 * h0
            data = self._interpolate(omch2, h0, neff)
        return {
            "r_s": data[0],
            "ks": self.ks,
            "pk_lin": data[1 : 1 + self.k_num],
            "pk_nl_0": data[1 + 1 * self.k_num : 1 + 2 * self.k_num],
            "pk_nl_z": data[1 + 2 * self.k_num :],
        }

    def _generate_data(self, savedata=True): # this function loops through the arrays on values for om, h0, neff etc. that we want to vary and saves the power spectra for each cosmology to an array - gets cosmo at z = 0 and 1 specified redshift 
        if self.vary_neff:
            self.logger.info(f"Generating CAMB data with {self.om_resolution} x {self.h0_resolution} x {self.neff_resolution}")
        else:
            self.logger.info(f"Generating CAMB data with {self.om_resolution} x {self.h0_resolution}")
            
        os.makedirs(self.data_dir, exist_ok=True)
        import camb

        pars = camb.CAMBparams()
        pars.set_dark_energy(w=-1.0, dark_energy_model="fluid")
        pars.InitPower.set_params(As=2.083e-9, ns=self.ns)
        pars.set_matter_power(redshifts=[self.redshift, 0.0], kmax=self.k_max)
        self.logger.info("Configured CAMB power and dark energy")

        data = np.zeros((self.om_resolution, self.h0_resolution, 1 + 3 * self.k_num))
        if self.vary_neff: 
            data = np.zeros((self.om_resolution, self.h0_resolution, self.neff_resolution, 1 + 3 * self.k_num))
            
        for i, omch2 in enumerate(self.omch2s):
            for j, h0 in enumerate(self.h0s):
                
                if self.vary_neff: 
                    for k, neff in enumerate(self.neffs):
                        
                        self.logger.info("Generating %d:%d:%d  %0.4f  %0.4f  %0.4f" % (i, j, k, omch2, h0, neff))
                        pars.set_cosmology(
                            H0=h0 * 100,
                            omch2=omch2,
                            mnu=self.mnu,
                            ombh2=self.omega_b * h0 * h0,
                            omk=0.0,
                            tau=0.066,
                            neutrino_hierarchy="degenerate",
                            num_massive_neutrinos=3,
                            nnu=neff, 
                        )
                        
                        pars.NonLinear = camb.model.NonLinear_none
                        results = camb.get_results(pars)
                        params = results.get_derived_params()
                        rdrag = params["rdrag"]
                        kh, z, pk_lin = results.get_matter_power_spectrum(minkh=self.k_min, maxkh=self.k_max, npoints=self.k_num)
                        pars.NonLinear = camb.model.NonLinear_pk
                        results.calc_power_spectra(pars)
                        kh, z, pk_nonlin = results.get_matter_power_spectrum(minkh=self.k_min, maxkh=self.k_max, npoints=self.k_num)
                        data[i, j, k, 0] = rdrag
                        data[i, j, k, 1 : 1 + self.k_num] = pk_lin[1, :]
                        data[i, j, k, 1 + self.k_num :] = pk_nonlin.flatten()
                        
                else:
                    
                    self.logger.info("Generating %d:%d  %0.4f  %0.4f" % (i, j, omch2, h0))
                    pars.set_cosmology(
                        H0=h0 * 100,
                        omch2=omch2,
                        mnu=self.mnu,
                        ombh2=self.omega_b * h0 * h0,
                        omk=0.0,
                        tau=0.066,
                        neutrino_hierarchy="degenerate",
                        num_massive_neutrinos=3,
                    )

                    pars.NonLinear = camb.model.NonLinear_none
                    results = camb.get_results(pars)
                    params = results.get_derived_params()
                    rdrag = params["rdrag"]
                    kh, z, pk_lin = results.get_matter_power_spectrum(minkh=self.k_min, maxkh=self.k_max, npoints=self.k_num)
                    pars.NonLinear = camb.model.NonLinear_pk
                    results.calc_power_spectra(pars)
                    kh, z, pk_nonlin = results.get_matter_power_spectrum(minkh=self.k_min, maxkh=self.k_max, npoints=self.k_num)
                    data[i, j, 0] = rdrag
                    data[i, j, 1 : 1 + self.k_num] = pk_lin[1, :]
                    data[i, j, 1 + self.k_num :] = pk_nonlin.flatten()

                    
        if savedata:
            self.logger.info(f"Saving to {self.filename}")
            np.save(self.filename, data)
        return data

    def interpolate(self, om, h0, neff, data=None):
        omch2 = (om - self.omega_b) * h0 * h0
        return self._interpolate(omch2, h0, neff, data=data)

    def _interpolate(self, omch2, h0, neff, data=None): 
        """Performs bilinear interpolation on the entire pk array - and extension of this to more variables if Neff is varied."""
        omch2_index = 1.0 * (self.om_resolution - 1) * (omch2 - self.omch2s[0]) / (self.omch2s[-1] - self.omch2s[0])

        # If omch2 == self.omch2s[-1] we can get an index out of bounds later due to rounding errors, so we
        # manually set the edge cases
        if omch2 >= self.omch2s[-1]:
            omch2_index = self.om_resolution - 1 - 1.0e-6

        if self.h0_resolution == 1:
            h0_index = 0
        else:
            h0_index = 1.0 * (self.h0_resolution - 1) * (h0 - self.h0s[0]) / (self.h0s[-1] - self.h0s[0])

            # If h0 == self.h0s[-1] we can get an index out of bounds later due to rounding errors, so we
            # manually set the edge cases
            if h0 == self.h0s[-1]:
                h0_index = self.h0_resolution - 1 - 1.0e-6
                
        if not self.vary_neff: 
            neff_index = 0
        else: 
            neff_index = 1.0 * (self.neff_resolution - 1) * (neff - self.neffs[0]) / (self.neffs[-1] - self.neffs[0])

            if neff == self.neffs[-1]: 
                neff_index = self.neff_resolution - 1 - 1.0e-6
                
        if data is None:
            data = self.data
            
        # code to do interpolation... ...before adding neff as an additional parameter which requiring interpolation 
        if not self.vary_neff:
            x = omch2_index - np.floor(omch2_index) # diff from index just below omch2 value (say x - x1), 1 - x_x1 gives x2 - x etc. 
            y = h0_index - np.floor(h0_index) # diff from index just below h0 value (say y - y1)

            v1 = data[int(np.floor(omch2_index)), int(np.floor(h0_index))]  # 00 - f(x,y,z) at x1, y1
            v2 = data[int(np.ceil(omch2_index)), int(np.floor(h0_index))]  # 01 - f(x,y,z) at x2, y1 

            if self.h0_resolution == 1:
                final = v1 * (1 - x) * (1 - y) + v2 * x * (1 - y)
            else:
                v3 = data[int(np.floor(omch2_index)), int(np.ceil(h0_index))]  # 10 - f(x,y,z) at x1, y2
                v4 = data[int(np.ceil(omch2_index)), int(np.ceil(h0_index))]  # 11 - f(x,y,z) at x2, y2 
                final = v1 * (1 - x) * (1 - y) + v2 * x * (1 - y) + v3 * y * (1 - x) + v4 * x * y 
            return final

        else:
            x_min_x1 = omch2_index - np.floor(omch2_index) # diff from index just below omch2 value (say x - x1), 1 - x_x1 gives x2 - x etc. 
            y_min_y1 = h0_index - np.floor(h0_index) # diff from index just below h0 value (say y - y1)
            z_min_z1 = neff_index - np.floor(neff_index) # diff from index just below neff value (say z - z1) 

            x2_min_x = 1 - x_min_x1
            y2_min_y = 1 - y_min_y1
            z2_min_z = 1 - z_min_z1
        
            f_111 = data[int(np.floor(omch2_index)), int(np.floor(h0_index)), int(np.floor(neff_index))]
            f_222 = data[int(np.ceil(omch2_index)), int(np.ceil(h0_index)), int(np.ceil(neff_index))]
            
            f_121 = data[int(np.floor(omch2_index)), int(np.ceil(h0_index)), int(np.floor(neff_index))]
            f_112 = data[int(np.floor(omch2_index)), int(np.floor(h0_index)), int(np.ceil(neff_index))]
            f_211 = data[int(np.ceil(omch2_index)), int(np.floor(h0_index)), int(np.floor(neff_index))]
            
            f_221 = data[int(np.ceil(omch2_index)), int(np.ceil(h0_index)), int(np.floor(neff_index))]
            f_122 = data[int(np.floor(omch2_index)), int(np.ceil(h0_index)), int(np.ceil(neff_index))]
            f_212 = data[int(np.ceil(omch2_index)), int(np.floor(h0_index)), int(np.ceil(neff_index))]
            
           
            if self.h0_resolution == 1 and self.om_resolution == 1: # only neff varies 
                
                final = f_111 * z2_min_z + f_112 * z_min_z1  # get weighted average of function given value of Neff desired 
                return final 
            
            elif self.h0_resolution == 1 and self.om_resolution > 1: # neff and om varies only 
                
                f_11z = f_111 * z2_min_z + f_112 * z_min_z1  # get weighted average of function given value of Neff desired over all Om
                f_21z = f_211 * z2_min_z + f_212 * z_min_z1
                
                f_x1z = x2_min_x * f_11z + x_min_x1 * f_21z # get weighted average of function given value of Om desired
                
                final = f_x1z 
                return final 
            
            elif self.h0_resolution > 1 and self.om_resolution == 1: # h0 and neff varies only 
        
                f_11z = f_111 * z2_min_z + f_112 * z_min_z1  # get weighted average of function given value of Neff desired over all h0
                f_12z = f_121 * z2_min_z + f_122 * z_min_z1
                
                f_1yz = y2_min_y * f_11z + y_min_y1 * f_12z # get weighted average of function given value of h0 desired
                
                final = f_1yz 
                return final 
            
            else: # vary all 3 parameters 
                
                f_11z = f_111 * z2_min_z + f_112 * z_min_z1 # get weighted average of function given value of Neff desired at om1, h01
                f_12z = f_121 * z2_min_z + f_122 * z_min_z1 # get weighted average of function given value of Neff desired at om1, h02
                
                f_21z = f_211 * z2_min_z + f_212 * z_min_z1 # get weighted average of function given value of Neff desired at om2, h01
                f_22z = f_221 * z2_min_z + f_222 * z_min_z1 # get weighted average of function given value of Neff desired at om2, h02
                
                f_1yz = y2_min_y * f_11z + y_min_y1 * f_12z # get weighted average of function given value of h0 desired at om1
                f_2yz = y2_min_y * f_21z + y_min_y1 * f_22z # get weighted average of function given value of h0 desired at om2
                
                f_xyz = x2_min_x * f_1yz + x_min_x1 * f_2yz # get weighted average of function given value of Om desired 
                
                final = f_xyz
                return final 
            

def test_rand_h0const():
    g = CambGenerator()
    g.load_data()

    def fn():
        g.get_data(np.random.uniform(0.1, 0.2))

    return fn


if __name__ == "__main__":

    import timeit
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)15s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

#     c = getCambGenerator(redshift=0.1, neff=3.044, h0_resolution=3, om_resolution=5)#, vary_neff=True, neff_resolution=3)
    
#     c._generate_data()

#     n = 10000
#     print("Takes on average, %.1f microseconds" % (timeit.timeit(test_rand_h0const(), number=n) * 1e6 / n))

#     plt.plot(c.ks, c.get_data(0.2, 0.75)["pk_lin"], color="b", linestyle="-", label=r"$\mathrm{Linear}\,\Omega_{m}=0.2\,h_0=0.75$")
#     plt.plot(c.ks, c.get_data(0.3, 0.75)["pk_lin"], color="r", linestyle="-", label=r"$\mathrm{Linear}\,\Omega_{m}=0.3\,h_0=0.75$")
#     plt.plot(c.ks, c.get_data(0.2, 0.75)["pk_nl_z"], color="b", linestyle="--", label=r"$\mathrm{Halofit}\,\Omega_{m}=0.2\,h_0=0.75$")
#     plt.plot(c.ks, c.get_data(0.3, 0.75)["pk_nl_z"], color="r", linestyle="--", label=r"$\mathrm{Halofit}\,\Omega_{m}=0.3\,h_0=0.75$")
    
#     plt.plot(c.ks, c.get_data(0.2, 0.70)["pk_lin"], color="b", linestyle="-.", label=r"$\mathrm{Linear}\,\Omega_{m}=0.2\,h_0=0.7$")
#     plt.plot(c.ks, c.get_data(0.3, 0.70)["pk_lin"], color="r", linestyle="-.", label=r"$\mathrm{Linear}\,\Omega_{m}=0.3\,h_0=0.7$")
#     plt.plot(c.ks, c.get_data(0.2, 0.70)["pk_nl_z"], color="b", linestyle=":", label=r"$\mathrm{Halofit}\,\Omega_{m}=0.2\,h_0=0.7$")
#     plt.plot(c.ks, c.get_data(0.3, 0.70)["pk_nl_z"], color="r", linestyle=":", label=r"$\mathrm{Halofit}\,\Omega_{m}=0.3\,h_0=0.7$")

#     plt.xscale("log")
#     plt.yscale("log")
#     plt.legend()
#     plt.savefig('test.png')
#     plt.show()
    
    
    c2 = getCambGenerator(redshift=0.1, neff=3.044, h0_resolution=3, om_resolution=3, vary_neff=True, neff_resolution=10)
    #c2._generate_data()
    
    #exit() 
    
    relpower = 1#c2.get_data(neff=3.0)["pk_lin"]
    plt.plot(c2.ks, c2.get_data(neff=2., h0=0.6, om=0.3)["pk_lin"]/relpower, color="b", linestyle="-", 
             label=r"$\mathrm{Linear}\,h_0=0.6\,\Omega_m=0.3\,$Neff=2.")
    plt.plot(c2.ks, c2.get_data(neff=3.5, h0=0.6, om=0.3)["pk_lin"]/relpower, color="g", linestyle=":", 
             label=r"$\mathrm{Linear}\,h_0=0.6\,\Omega_m=0.3\,$Neff=3.5")
    plt.plot(c2.ks, c2.get_data(neff=2., h0=0.7, om=0.3)["pk_lin"]/relpower, color="r", linestyle="-.", 
             label=r"$\mathrm{Linear}\,h_0=0.7\,\Omega_m=0.3\,$Neff=2")
    plt.plot(c2.ks, c2.get_data(neff=3.5, h0=0.7, om=0.3)["pk_lin"]/relpower, color="y", linestyle="--", 
             label=r"$\mathrm{Linear}\,h_0=0.7\,\Omega_m=0.3\,$Neff=3.5")
    
    plt.plot(c2.ks, c2.get_data(neff=2., h0=0.6, om=0.2)["pk_lin"]/relpower, color="b", linestyle="-", 
             label=r"$\mathrm{Linear}\,h_0=0.6\,\Omega_m=0.2\,$Neff=2.")
    plt.plot(c2.ks, c2.get_data(neff=3.5, h0=0.6, om=0.2)["pk_lin"]/relpower, color="g", linestyle=":", 
             label=r"$\mathrm{Linear}\,h_0=0.6\,\Omega_m=0.2\,$Neff=3.5")
    plt.plot(c2.ks, c2.get_data(neff=2., h0=0.7, om=0.2)["pk_lin"]/relpower, color="r", linestyle="-.", 
             label=r"$\mathrm{Linear}\,h_0=0.7\,\Omega_m=0.2\,$Neff=2")
    plt.plot(c2.ks, c2.get_data(neff=3.5, h0=0.7, om=0.2)["pk_lin"]/relpower, color="y", linestyle="--", 
             label=r"$\mathrm{Linear}\,h_0=0.7\,\Omega_m=0.2\,$Neff=3.5")
    
    #plt.plot(c2.ks, c2.get_data(neff=4.0)["pk_lin"]/relpower, color="k", linestyle="--", label=r"$\mathrm{Linear}\,$Neff=4")
    
#     plt.plot(c2.ks, c2.get_data(neff=2.5)["pk_nl_z"]/relpower, color="b", linestyle="--", label=r"$\mathrm{Halofit}\,$Neff=2.5")
#     plt.plot(c2.ks, c2.get_data(neff=3.5)["pk_nl_z"]/relpower, color="r", linestyle="--", label=r"$\mathrm{Halofit}\,$Neff=3.5")
    
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig('test2.png')
    plt.show()
    
    
#     c1 = getCambGenerator(neff=3.045, h0_resolution=3, om_resolution=1)
#     c1._generate_data()

#     plt.plot(c1.ks, c1.get_data()["pk_lin"], color="b", linestyle="-")
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.legend()
#     plt.savefig('test1.png')
#     plt.show()