import sys
sys.path.append('/home1/yxiao977/labwork/Akita_dinoflagellate')
import data_util
import cooler
import logging
import pandas as pd
import cooltools
import numpy as np

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s')
logger = logging.getLogger('test_func')
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Disable propagation to the root logger

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
# logger.disabled = True



# create a small toy example inside the test func with a human readable small data
def test_cis_total_filter(thres, clr_test=None):

    if clr_test == None:
        logger.debug('Start to generate a toy example...')
        # Generate a toy example
        bins=pd.DataFrame(
            [["chr1", 0, 5],
                ["chr1", 5, 8],
                ["chrX", 0, 5],
                ["chrX", 5, 10]],
        columns=["chrom", "start", "end"],
        )

        # only has contacts within the same chromsomes
        # only has contacts with the different chromosomes
        # > 0.5 cis-total ratio
        # = 0.5 cis-total ratio
        pixels = pd.DataFrame(
            [[0, 0, 6], 
            [1, 2, 3], [1, 3, 35],
            [2, 2, 3], [2, 3, 30],
            [3, 3, 5]], 
            columns=["bin1_id", "bin2_id", "count"]
        )

    clr_file = "/home1/yxiao977/sc1/train_akita/test_data/test_data_util.cool"
    cooler.create_cooler(clr_file, bins, pixels)
    clr_test = cooler.Cooler(clr_file)
    cooler.balance_cooler(clr_test, ignore_diags=0, store=True, store_name='weight')
    clr_test = cooler.Cooler(clr_file)


    logger.debug('Start to test filter_pixels function...')
    output_path = f"/home1/yxiao977/sc1/train_akita/test_data/test_data_util_{thres}filtered.cool"
    data_util.filter_pixels_by_masked_bin(clr_test, thres, output_path, bins_filters=None, nproc=16, chunksize=10_000_000)
    clr_filtered = cooler.Cooler(output_path)

    # check if clr_filtered has nan cis_total ratio for bins filtered out
    logger.debug('Start to examine the result...')
    coverage = cooltools.coverage(clr_test, ignore_diags=0)
    cis_total_cov = coverage[0] / coverage[1]
    bins_mask = cis_total_cov <= thres

    coverage_filtered = cooltools.coverage(clr_filtered, ignore_diags=0)
    cis_total_cov_filtered = coverage_filtered[0] / coverage_filtered[1]
    cis_total_cov_filtered_bad = cis_total_cov_filtered[bins_mask]

    assert np.isnan(cis_total_cov_filtered_bad).all()
    
    # check if good bins has nan cis-total ratio
    logger.info('\n\n######### Pass the test #########')


    
    
    
    # logger.debug('Start to sum up counts')
    # check_nan = lambda x: np.isnan(x['count']).all()
    
    # check_result = np.sum(list(map(check_nan, bad_pixels_binid1)))
    # assert check_result == len(bad_pixels_binid1)
    # logger.info('Pass bin1 checkpoint')


    # check_result = np.sum(list(map(check_nan, bad_pixels_binid2)))
    # assert check_result == len(bad_pixels_binid2)
    # logger.info('Pass bin2 checkpoint')


