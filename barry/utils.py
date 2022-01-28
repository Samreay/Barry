import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return average, np.sqrt(variance)


def weighted_avg_and_cov(values, weights, axis):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis=axis)
    covariance = np.cov(values, aweights=weights, rowvar=axis)
    return average, covariance


def break_vector_and_get_blocks(x, number_breaks, keep_indices):
    return np.array(np.split(x, number_breaks))[keep_indices, :].flatten()


def break_matrix_and_get_blocks(matrix, number_breaks, keep_indices):
    x = break2d_into_blocks(matrix, number_breaks)
    reduced = x[keep_indices, :, :, :][:, keep_indices, :, :]
    result = stitch_blocks_together(reduced)
    return result


def break2d_into_blocks(x, number_breaks):
    num_elem = int(np.sqrt(x.size / (number_breaks ** 2)))
    return x.reshape((number_breaks, num_elem, number_breaks, num_elem)).transpose(0, 2, 1, 3)


def stitch_blocks_together(x):
    n = int(np.sqrt(x.size))
    return x.transpose(0, 2, 1, 3).reshape((n, n))


def create_histogram_plot():
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")


def get_hpc():
    hpc = os.environ.get("HPC")
    logging.debug(f"HPC environment key is {hpc}")
    return hpc


def get_model_comparison_dataframe(fitter):
    """Uses fitter.load to create a comparison dataframe on the first column of fitter results (presumed to be alpha)

    Will only produce a row if a given realisation has a successful fit for all models.

    Returns
    -------
    model_results : dict of pd.DataFrame
        the model name linked to a dataframe containing the avg alpha, std alpha, max alpha, max log posterior and realisation number for each realisation
    summary : pd.DataFrame
        One row per model, giving the mean alpha over all realisations, mean of the alpha stds (mean reported error), and std of the mean alpha (scatter in reported mean)

    """
    model_results = {}
    for posterior, weight, chain, evidence, model, data, extra in fitter.load():
        n = extra["name"]
        if model_results.get(n) is None:
            model_results[n] = []
        i = posterior.argmax()
        a, s = weighted_avg_and_std(chain[:, 0], weights=weight)
        model_results[n].append([a, s, chain[i, 0], posterior[i], extra["realisation"]])

    for label in model_results.keys():
        model_results[label] = pd.DataFrame(model_results[label], columns=["avg", "std", "max", "posterior", "realisation"])

    # This shouldnt be necessary, but if a job or two gets killed or a node restarted, this will remove that realisation
    all_ids = pd.concat(tuple([m[["realisation"]] for m in model_results.values()]))
    counts = all_ids.groupby("realisation").size().reset_index()
    max_count = counts.values[:, 1].max()
    good_ids = counts.loc[counts.values[:, 1] == max_count, ["realisation"]]
    num_dropped = (counts.values[:, 1] != max_count).sum()
    if num_dropped:
        logging.warning(
            f"Warning, {num_dropped} realisations did not have a fit for all models."
            + " Rerun with  fitter = Fitter(dir_name, remove_output=False) to fill in the missing fits"
        )

    # Merge results
    summary = []
    for label, df in model_results.items():
        model_results[label] = pd.merge(good_ids, df, how="left", on="realisation")
        summary.append(
            [label, np.mean(model_results[label]["avg"]), np.mean(model_results[label]["std"]), np.std(model_results[label]["avg"])]
        )
    summary = pd.DataFrame(summary, columns=["Name", "Mean mean", "Mean std", "Std mean"])

    return model_results, summary
