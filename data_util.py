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
    if not isinstance(filter_lst, list):
        logger.error('filter_lst parameter takes a list')
        
    bin_mask = np.array([True] * clr.bins().shape[0])
    for i, filter in enumerate(filter_lst):
        bin_mask *= filter(clr, thres_lst[i])
    
    return bin_mask

### pass only good bins index  rather than bins_table_size, bin_mask
def _pixel_filter(chunk_pixels, good_bins_index):

    pixels_mask = chunk_pixels['bin1_id'].isin(good_bins_index) * chunk_pixels['bin2_id'].isin(good_bins_index)
    return chunk_pixels[pixels_mask]

        
def pixel_iter_chunks(clr, chunksize):

    selector = clr.pixels()
    for lo, hi in cooler.util.partition(0, len(selector), chunksize):
        chunk = selector[lo:hi]
        yield chunk

#### try to make a decorator for the nproc if statement 
def pool_decorator(func, map_param, nproc):

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

# output_uri, clr, has a separate bin filter function that accept filters sequentially applied to the bin matrix return a bin mask
def create_filtered_cooler(output_uri, clr, bin_mask, nproc=16, chunksize=10_000_000):

    logger.debug('Start to create cooler file...')
    bin_table = clr.bins()[:]
    good_bins_index = np.array(range(clr.bins().shape[0]))[bin_mask]
    pixels_filter = functools.partial(_pixel_filter, good_bins_index=good_bins_index)

    pool_create_cooler = pool_decorator(cooler.create_cooler, map_param='pixels', nproc=nproc)
    pool_create_cooler(pixels_filter, pixel_iter_chunks(clr, chunksize), output_uri, bins=bin_table, ordered=True, columns=['count'])


    logger.debug('done')


