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

    
def _bins_cis_total_ratio_filter(clr, thres):
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
    bins_mask = cis_total_cov > thres

    return bins_mask

### double check if filter work correctly
def _pixels_filter(chunk_pixels, bins_table_size, bins_mask):
    """
    Filter out pixels that belong to bad bins.

    Parameters:
    -----------
    bin_table : numpy.ndarray
        A 2D array of shape (n_bins, n_features) containing the features of each bin.
    chunk_pixels : pandas.DataFrame
        A DataFrame containing the pixels to filter.
    bins_mask : numpy.ndarray
        A boolean array of shape (n_bins,) indicating which bins are bad.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing only the pixels that belong to good bins.
    """
    good_bins_index = np.array(range(bins_table_size))[bins_mask]
    pixels_mask = chunk_pixels['bin1_id'].isin(good_bins_index) * chunk_pixels['bin2_id'].isin(good_bins_index)
    return chunk_pixels[pixels_mask]

def pixels_filter_generator(bins_table_size, bins_mask):
    """
    Returns a partial function that filters pixels based on the provided bins table and bin mask.

    Parameters:
    -----------
    bins_table_size : int
        A integer value indicates the size of bin table.
    bins_mask : numpy.ndarray
        A numpy array containing the bin mask.

    Returns:
    --------
    partial function
        A partial function that filters pixels based on the provided bins table and bin mask.
    """

    return functools.partial(_pixels_filter, bins_table_size=bins_table_size, bins_mask=bins_mask)

class PixelsIterator:
    def __init__(self, clr_pixels_selector, pixels_size, chunksize):
        self.chunksize = chunksize
        self.max = pixels_size
        self.pixels = clr_pixels_selector

    def __iter__(self):
        self.pivot = 0
        return self

    def __next__(self):
        if (self.pivot + self.chunksize) < self.max:
            pivot = self.pivot
            self.pivot += self.chunksize
            return self.pixels[pivot : self.pivot]
        elif self.pivot < self.max:
            pivot = self.pivot
            self.pivot = self.max
            return self.pixels[pivot : self.pivot]
        else:
            raise StopIteration

def filter_pixels_by_masked_bin(clr, thres, output_path, bins_filters=None, nproc=16, chunksize=10_000_000):
    """
    Filter pixels of a cooler object based on a binary mask of genomic bins.

    Parameters
    ----------
    clr : cooler.Cooler
        A cooler object containing contact matrices and genomic bin information.
    thres : float
        A threshold value for filtering genomic bins based on their cis/trans ratio.
    output_path : str
        The path to the output cooler file.
    bins_filters : list of functions, optional
        A list of functions that generate binary masks for genomic bins. Default is None.
    nproc : int, optional
        The number of processes to use for parallelization. Default is 16.
    chunksize : int, optional
        The number of pixels to process at a time. Default is 10,000,000.

    Returns
    -------
    None

    Notes
    -----
    This function creates a binary mask of genomic bins based on their cis/trans ratio,
    and uses it to filter the pixels of the input cooler object. The resulting filtered
    pixels are written to a new cooler file at the specified output path.

    """
    if bins_filters == None:
        bins_filters = [_bins_cis_total_ratio_filter]

    for bins_filter in bins_filters:
        logger.info(f'Start to make bin mask with {thres} threshold...')
        bins_mask = bins_filter(clr, thres)
        bins_table = clr.bins()[:]
        tot_pixels = clr.pixels().shape[0]

        logger.debug('Start to create pixels counts file...')
        pixels_chunks = PixelsIterator(clr.pixels(), tot_pixels, chunksize)
        pixels_filter = pixels_filter_generator(bins_table.shape[0], bins_mask)

        logger.debug('Start to create cooler file...')
        p = Pool(nproc)
        cooler.create_cooler(output_path, bins=bins_table, pixels=p.map(pixels_filter, pixels_chunks), ordered=True, columns=['count'])
        p.close()
        logger.debug('done')
    