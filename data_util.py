import numpy as np
import pandas as pd
import cooltools
import cooler
import functools
from multiprocessing import Pool

import logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s')
logger = logging.getLogger('data_util')
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Disable propagation to the root logger

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
 
def cis_total_ratio_filter(clr, thres):
    """
    Filter out bins with low cis-to-total coverage ratio from a Cooler object.

    Parameters
    ----------
    clr : cooler.Cooler
        A Cooler object containing Hi-C contact matrices.
    thres : float
        The threshold cis-to-total coverage ratio below which bins are considered bad.

    Returns
    -------
    numpy.ndarray
        An array of bin mask.
    """
    coverage = cooltools.coverage(clr)
    cis_total_cov = coverage[0] / coverage[1]
    bin_mask = cis_total_cov > thres

    return bin_mask

def generate_bin_mask(clr, filter_lst, thres_lst):
    """
    Generates a binary mask for a given `clr` object based on a list of filters and thresholds.

    Parameters:
    -----------
    clr : cooler.Cooler
        A cooler object containing Hi-C contact matrices.
    filter_lst : list
        A list of filter functions to apply to the contact matrices.
    thres_lst : list
        A list of thresholds to use for each filter function.

    Returns:
    --------
    bin_mask : numpy.ndarray
        A binary mask indicating which genomic bins pass all filters.
    """
    if not isinstance(filter_lst, list):
        logger.error('filter_lst parameter takes a list')
        
    bin_mask = np.array([True] * clr.bins().shape[0])
    for i, filter in enumerate(filter_lst):
        bin_mask *= filter(clr, thres_lst[i])
    
    return bin_mask

def _pixel_filter(chunk_pixels, good_bins_index):
    """
    Filters a chunk of pixels based on a list of good bin indices.

    Parameters
    ----------
    chunk_pixels : pandas.DataFrame
        A DataFrame containing the pixels to be filtered. It must have columns 'bin1_id' and 'bin2_id'.
    good_bins_index : list of int
        A list of indices representing the good bins.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing only the pixels whose bin1_id and bin2_id are in good_bins_index.
    """
    
    pixels_mask = chunk_pixels['bin1_id'].isin(good_bins_index) * chunk_pixels['bin2_id'].isin(good_bins_index)
    return chunk_pixels[pixels_mask]

def pixel_iter_chunks(clr, chunksize):
    """
    Iterate over the pixels of a cooler object in chunks of a given size.

    Parameters:
    -----------
    clr : cooler.Cooler
        A cooler object containing Hi-C data.
    chunksize : int
        The size of each chunk of pixels to iterate over.

    Yields:
    -------
    chunk : numpy.ndarray
        A chunk of pixels of size `chunksize`.
    """
    selector = clr.pixels()
    for lo, hi in cooler.util.partition(0, len(selector), chunksize):
        chunk = selector[lo:hi]
        yield chunk

def pool_decorator(func, map_param, nproc):
    """
    A decorator function that parallelizes the execution of a given function using multiprocessing.Pool.

    Parameters:
    -----------
    func : function
        The function to be parallelized.
    map_param : str
        The name of the parameter in func that should be mapped over.
    nproc : int
        The number of processes to use for parallelization.

    Returns
    -------
    decorated_func : function
        The decorated function that parallelizes the execution of func.
    """
    def decorated_func(func_in_map, iter_chunks, *args, **kwargs):

        if nproc > 1:
            pool = Pool(nproc)
            mymap = pool.map
        else:
            mymap = map

        func(*args, **kwargs, **{map_param: mymap(func_in_map, iter_chunks)})
        
        if nproc > 1: 
            pool.close()

    return decorated_func

def create_filtered_cooler(output_uri, clr, bin_mask, nproc=16, chunksize=10_000_000):
    """
    Create a filtered cooler file from a given cooler object and a binary mask of good bins.

    Parameters
    ----------
    output_uri : str
        The URI of the output cooler file to be created.
    clr : cooler.Cooler
        The cooler object to be filtered.
    bin_mask : numpy.ndarray
        A boolean array indicating which bins to keep (True) and which to discard (False).
        Must have the same length as the number of bins in the cooler object.
    nproc : int, optional
        The number of processes to use for parallelization. Default is 16.
    chunksize : int, optional
        The number of pixels to process per chunk. Default is 10,000,000.

    Returns
    -------
    None
    """
    logger.debug('Start to create cooler file...')
    bin_table = clr.bins()[:]
    good_bins_index = np.array(range(clr.bins().shape[0]))[bin_mask]
    pixels_filter = functools.partial(_pixel_filter, good_bins_index=good_bins_index)

    pool_create_cooler = pool_decorator(cooler.create_cooler, map_param='pixels', nproc=nproc)
    pool_create_cooler(pixels_filter, pixel_iter_chunks(clr, chunksize), output_uri, bins=bin_table, ordered=True, columns=['count'])


    logger.debug('done')


