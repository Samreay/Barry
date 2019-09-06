import logging
from barry.models import CorrSeo2016

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    from barry.datasets import CorrelationFunction_SDSS_DR12_Z061_NGC

    for recon in [True, False]:
        model = CorrSeo2016(recon=recon)
        model_smooth = CorrSeo2016(recon=recon, smooth=True)
        model.set_default("om", 0.31)
        model_smooth.set_default("om", 0.31)
        # Assuming the change from 0.675 to 0.68 is something we can ignore, or we can add h0 to the default parameters.

        dataset1 = CorrelationFunction_SDSS_DR12_Z061_NGC(recon=recon, reduce_cov_factor=10)
        data1 = dataset1.get_data()

        # First comparison - the actual recon data
        model.set_data(data1)
        p, minv = model.optimize()
        model_smooth.set_data(data1)
        p2, minv2 = model_smooth.optimize()
        print(p)
        print(minv)
        model.plot(p, smooth_params=p2)

        # FINDINGS
        # 0.999 and 1.001
