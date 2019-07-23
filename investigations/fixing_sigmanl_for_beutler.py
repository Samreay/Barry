import logging

from barry.framework.models import PowerBeutler2017

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    recon = True
    from barry.framework.datasets.mock_power import MockSDSSPowerSpectrum
    dataset1 = MockSDSSPowerSpectrum(recon=recon)
    data1 = dataset1.get_data()

    # Run on mean
    model = PowerBeutler2017(recon=recon, name=f"Beutler2017, recon={recon}")
    model.set_data(data1)
    p, minv = model.optimize(maxiter=100)
    sigma_nl = p["sigma_nl"]

    print(f"Sigma_nl found to be {sigma_nl:0.3f}")
    model.set_default("sigma_nl", sigma_nl)
    model.set_fix_params(["om", "sigma_nl"])

    p2, minv2 = model.optimize(maxiter=100)
    print(p, minv)
    print(p2, minv2)
