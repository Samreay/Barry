import logging

from barry.models import PowerDing2018

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    recon = True
    model1 = PowerDing2018(recon=recon, name=f"Ding2018, recon={recon}")
    model_smooth = PowerDing2018(recon=recon, name=f"Ding2018, recon={recon}", smooth=True)

    from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
    from barry.datasets import DummyPowerSpectrum_SDSS_DR12_Z061_NGC

    dataset1 = PowerSpectrum_SDSS_DR12_Z061_NGC(name="Recon mean", recon=recon)
    dataset2 = DummyPowerSpectrum_SDSS_DR12_Z061_NGC(name="Dummy data, real window fn", dummy_window=False)
    dataset3 = DummyPowerSpectrum_SDSS_DR12_Z061_NGC(name="DummyWindowFnToo", dummy_window=True)
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
