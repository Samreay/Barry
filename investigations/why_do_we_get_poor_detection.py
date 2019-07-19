import logging

from barry.framework.models import PowerBeutler2017

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    recon = True
    model1 = PowerBeutler2017(recon=recon, name=f"Beutler2017, recon={recon}")
    model_smooth = PowerBeutler2017(recon=recon, name=f"Beutler2017, recon={recon}", smooth=True)

    from barry.framework.datasets.mock_power import MockSDSSPowerSpectrum
    i = 12
    dataset1 = MockSDSSPowerSpectrum(name=f"Realisation {i}", recon=recon, average=False, realisation=i)

    data1 = dataset1.get_data()

    # First comparison - the actual recon data
    model1.set_data(data1)
    p, minv = model1.optimize()
    model_smooth.set_data(data1)
    p2, minv2 = model_smooth.optimize()
    print(p, p2)
    print(minv, minv2)
    chi2 = -2 * (minv - minv2)
    print("chi2 is ", chi2)
    model1.plot(p, smooth_params=p2)
