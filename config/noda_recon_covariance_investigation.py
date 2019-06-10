import sys


sys.path.append("..")
from investigations.does_nova_cov_match_bruteforce_mixed import calc_cov_noda
from barry.setup import setup
from barry.framework.models import PowerNoda2019
from barry.framework.datasets import MockPowerSpectrum
from barry.framework.postprocessing import BAOExtractor, PureBAOExtractor
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.samplers.ensemble import EnsembleSampler
from barry.framework.fitter import Fitter

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = CambGenerator()
    r_s, _ = c.get_data()

    postprocess = BAOExtractor(r_s, reorder=False)
    r = True
    models = [PowerNoda2019(postprocess=postprocess, recon=r, name="")]

    datas = [
        MockPowerSpectrum(name="Mock covariance", recon=r, min_k=0.03, max_k=0.30, postprocess=postprocess, apply_hartlap_correction=False),
        MockPowerSpectrum(name="Nishimichi, full", recon=r, min_k=0.03, max_k=0.30, postprocess=postprocess, apply_hartlap_correction=False),
        MockPowerSpectrum(name="Nishimichi, diag", recon=r, min_k=0.03, max_k=0.30, postprocess=postprocess, apply_hartlap_correction=False),
    ]

    # Compute the pseudo-analytic cov from Noda and Nishimichi
    data = MockPowerSpectrum(recon=r, min_k=0.03, max_k=0.30, apply_hartlap_correction=False)
    data2 = MockPowerSpectrum(recon=r, min_k=0.03, max_k=0.30, apply_hartlap_correction=False, fake_diag=True)
    ks = data.ks
    pk = data.data
    pk_cov = data.cov
    pk_cov_diag = data2.cov
    e = PureBAOExtractor(r_s)
    denoms = e.postprocess(ks, pk, return_denominator=True)
    k_range = e.get_krange()
    is_extracted = postprocess.get_is_extracted(ks)
    cov_noda = calc_cov_noda(pk_cov, denoms, ks, pk, k_range, is_extracted)
    cov_noda_diag = calc_cov_noda(pk_cov_diag, denoms, ks, pk, k_range, is_extracted)

    # Override the mock covariance in the second model with the newly computed one
    datas[1].set_cov(cov_noda, apply_correction=False)
    datas[2].set_cov(cov_noda_diag, apply_correction=False)

    sampler = EnsembleSampler(temp_dir=dir_name)
    fitter = Fitter(dir_name)
    fitter.set_models(*models)
    fitter.set_data(*datas)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(20)
    fitter.fit(file, viewer=False)

    if fitter.is_laptop():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, model, data in fitter.load():
            name = f"{model.get_name()} {data.get_name()}"
            linestyle = "--" if "FitOm" in name else "-"
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), name=name, linestyle=linestyle)
        c.configure(shade=True, bins=20)
        c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        with open(pfn + "_params.txt", "w") as f:
            f.write(c.analysis.get_latex_table(transpose=True))


