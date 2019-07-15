import logging

from barry.framework.models import PowerBeutler2017

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    from barry.framework.datasets.mock_power import MockSDSSPowerSpectrum

    for recon in [True, False]:
        model = PowerBeutler2017(recon=recon, name=f"Beutler2017, recon={recon}")
        model_smooth = PowerBeutler2017(recon=recon, name=f"Beutler2017, recon={recon}", smooth=True)

        model.set_default("om", 0.31)
        model_smooth.set_default("om", 0.31)
        # Assuming the change from 0.675 to 0.68 is something we can ignore, or we can add h0 to the default parameters.

        dataset1 = MockSDSSPowerSpectrum(name=f"SDSS recon={recon}", recon=recon, reduce_cov_factor=31.62)
        data1 = dataset1.get_data()

        # First comparison - the actual recon data
        model.set_data(data1)
        p, minv = model.optimize()
        model_smooth.set_data(data1)
        p2, minv2 = model_smooth.optimize()
        print(p)
        print(minv)
        model.plot(p, smooth_params=p2)
