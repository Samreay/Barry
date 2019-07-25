import logging

from barry.framework.models import PowerBeutler2017

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    recon = True
    model1 = PowerBeutler2017(recon=recon, name=f"Beutler2017, recon={recon}")
    model_smooth = PowerBeutler2017(recon=recon, name=f"Beutler2017, recon={recon}", smooth=True)

    from barry.framework.datasets.mock_power import PowerSpectrum_SDSS_DR12_Z051_NGC
    from barry.framework.datasets.dummy_power import DummyPowerSpectrumSDSSDR12Z051NGC
    dataset1 = PowerSpectrum_SDSS_DR12_Z051_NGC(recon=recon)
    dataset2 = DummyPowerSpectrumSDSSDR12Z051NGC(name="Dummy data, real window fn", dummy_window=False)
    dataset3 = DummyPowerSpectrumSDSSDR12Z051NGC(name="DummyWindowFnToo", dummy_window=True)
    data1 = dataset1.get_data()
    data2 = dataset2.get_data()
    data3 = dataset3.get_data()

    # First comparison - the actual recon data
    model1.set_data(data1)
    p, minv = model1.optimize()
    model_smooth.set_data(data1)
    p2, minv2 = model_smooth.optimize()
    print(p)
    print(minv)
    model1.plot(p, smooth_params=p2)

    # The second comparison, dummy data with real window function
    model1.set_data(data2)
    p, minv = model1.optimize()
    model_smooth.set_data(data2)
    p2, minv2 = model_smooth.optimize()
    print(p)
    print(minv)
    model1.plot(p, smooth_params=p2)

    # Dummy data *and* dummy window function
    model1.set_data(data3)
    p, minv = model1.optimize()
    model_smooth.set_data(data3)
    p2, minv2 = model_smooth.optimize()
    print(p)
    print(minv)
    model1.plot(p, smooth_params=p2)

    # FINDINGS
    # Yes, Beutler2017 can thankfully fit pklin perfectly.