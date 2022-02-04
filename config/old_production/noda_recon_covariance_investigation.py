import sys

sys.path.append("..")
from investigations.does_noda_cov_match_bruteforce_mixed import calc_cov_noda_mixed
from barry.config import setup
from barry.models import PowerNoda2019
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.postprocessing import BAOExtractor, PureBAOExtractor
from barry.cosmology.camb_generator import getCambGenerator
from barry.samplers import DynestySampler
from barry.fitter import Fitter
import numpy as np

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = getCambGenerator()
    r_s = c.get_data()["r_s"]

    postprocess = BAOExtractor(r_s)
    r = True
    model = PowerNoda2019(postprocess=postprocess, recon=r, name="")
    mink = 0.03
    maxk = 0.30
    datas = [
        PowerSpectrum_SDSS_DR12_Z061_NGC(name="Mock covariance", recon=r, min_k=mink, max_k=maxk, postprocess=postprocess),
        PowerSpectrum_SDSS_DR12_Z061_NGC(name="Nishimichi, full", recon=r, min_k=mink, max_k=maxk, postprocess=postprocess),
        PowerSpectrum_SDSS_DR12_Z061_NGC(name="Nishimichi, diag", recon=r, min_k=mink, max_k=maxk, postprocess=postprocess),
    ]

    # Compute the pseudo-analytic cov from Noda and Nishimichi
    data = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, min_k=0.0, max_k=0.32)
    data2 = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, min_k=0.0, max_k=0.32, fake_diag=True)
    ks = data.ks
    pk = data.data
    pk_cov = data.cov
    pk_cov_diag = data2.cov
    e = PureBAOExtractor(r_s)
    denoms = e.postprocess(ks, pk, None, return_denominator=True)
    k_range = e.get_krange()
    is_extracted = postprocess.get_is_extracted(ks)
    cov_original = datas[0].cov
    cov_noda = calc_cov_noda_mixed(pk_cov, denoms, ks, pk, k_range, is_extracted)
    cov_noda_diag = calc_cov_noda_mixed(pk_cov_diag, denoms, ks, pk, k_range, is_extracted)

    # trim covariance to the same shape as the data
    mask = (ks >= mink) & (ks <= maxk)
    mask2d = mask[np.newaxis, :] & mask[:, np.newaxis]
    cov_noda = cov_noda[mask2d].reshape((mask.sum(), mask.sum()))
    cov_noda_diag = cov_noda_diag[mask2d].reshape((mask.sum(), mask.sum()))

    # redorder cov matrix, pk goes first, then the extracted part
    mask_extracted = postprocess.get_is_extracted(datas[0].ks)
    index_sort = np.arange(mask_extracted.size).astype(np.int) + (1000 * mask_extracted)
    a = np.argsort(index_sort)
    cov_noda = cov_noda[:, a]
    cov_noda = cov_noda[a, :]
    cov_noda_diag = cov_noda_diag[:, a]
    cov_noda_diag = cov_noda_diag[a, :]

    datas[1].set_cov(cov_noda)
    datas[2].set_cov(cov_noda_diag)

    sampler = DynestySampler(temp_dir=dir_name)
    fitter = Fitter(dir_name)
    fitter.add_model_and_dataset(model, datas[0], name="Mock-computed")
    fitter.add_model_and_dataset(model, datas[1], name="Nishimichi 2018, full")
    fitter.add_model_and_dataset(model, datas[2], name="Nishimichi 2018, diagonal")
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.should_plot():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=20)
        c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 0.9982})
        c.plotter.plot(filename=[pfn + "_contour2.png", pfn + "_contour2.pdf"], parameters=2, truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 0.9982})
        c.plotter.plot_summary(filename=pfn + "_summary.png", truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 0.9982}, errorbar=True)
        c.plotter.plot_summary(
            filename=pfn + "_summary2.png", truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 0.9982}, errorbar=True, parameters=1, extra_parameter_spacing=0.5
        )
        c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 0.9982})
        with open(pfn + "_params.txt", "w") as f:
            f.write(c.analysis.get_latex_table(transpose=True))

    # FINDINGS
    # Analytic derivation works well, so long as we dont assume independence.
