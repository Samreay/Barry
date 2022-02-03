from abc import ABC
import logging


class Dataset(ABC):
    """Abstract Dataset class.

    Concrete implementations need to implement the `get_data` method, which
    should return a length 1 array with the sole element being a dictionary
    of data.

    Attributes
    ----------
    name : str
        The name of the dataset
    logger : `logging.logger`
        Class logger

    """

    def __init__(self, name, isotropic=True, recon=None, realisation=None):
        self.recon = recon
        self.isotropic = isotropic
        self.realisation = realisation
        self.name = name
        self.logger = logging.getLogger("barry")

    def get_name(self):
        return self.name

    def get_data(self):
        """Return a list of data dictionaries

        Returns
        -------
        datas : list
            A list of dictionaries. For a single data source, this will a list one element long.
        """
        raise NotImplementedError("Please implement get_data")


class MultiDataset(Dataset, ABC):
    """Dataset wrapping multiple datasets. Used for combining *independent* datasets.

    Attributes
    ----------
    datas : list
        List of `Dataset` objects.

    """

    def __init__(self, name, datasets):
        super().__init__(name)
        self.datasets = datasets

    def get_data(self):
        """Returns a flattened list of data from each child dataset.

        Returns
        -------
        datasets : list
            A list of datasets
        """
        return [i for d in self.datasets for i in d.get_data()]


if __name__ == "__main__":
    print("Calling a Generic model class as main does not do anything. Try running the Concrete classes: ")
    print("dataset_power_spectrum.py")
    print("dataset_correlation_function.py")
