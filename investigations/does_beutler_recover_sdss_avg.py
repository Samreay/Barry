import logging

from barry.framework.models import PowerBeutler2017

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    from barry.framework.datasets.mock_power import MockPowerSpectrum

    for recon in [False]:
        model1 = PowerBeutler2017(recon=recon, name=f"Beutler2017, recon={recon}")
        model_smooth = PowerBeutler2017(recon=recon, name=f"Beutler2017, recon={recon}", smooth=True)

        dataset1 = MockPowerSpectrum(name=f"SDSS recon={recon}", recon=recon, min_k=0.02, max_k=0.3, reduce_cov_factor=30, step_size=20, data_dir="sdss_mgs_mocks")
        data1 = dataset1.get_data()

        # First comparison - the actual recon data
        model1.set_data(data1)
        p, minv = model1.optimize()
        model_smooth.set_data(data1)
        p2, minv2 = model_smooth.optimize()
        print(p)
        print(minv)
        model1.plot(p, smooth_params=p2)
