import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
import data_util
import cooler
import logging
import pandas as pd
import cooltools
import numpy as np

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s")
logger = logging.getLogger("test_data_util")
logger.propagate = False  # Disable propagation to the root logger

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

def _generate_test_example():
    logger.debug("Start to generate a toy example...")
    # Generate a toy example
    bins = pd.DataFrame(
        [["chr1", 0, 5], 
        ["chr1", 5, 8], 
        ["chrX", 0, 5], 
        ["chrX", 5, 10]
        ],
        columns=["chrom", "start", "end"],
    )

    # only has contacts within the same chromsomes
    # only has contacts with the different chromosomes
    # > 0.5 cis-total ratio
    # = 0.5 cis-total ratio
    pixels = pd.DataFrame(
        [[0, 0, 6], 
        [1, 2, 3], [1, 3, 35], 
        [2, 2, 3], [2, 3, 35]
        ],
        columns=["bin1_id", "bin2_id", "count"],
    )
    clr_file = "./test_data_util.cool"
    cooler.create_cooler(clr_file, bins, pixels)
    clr_test = cooler.Cooler(clr_file)
    cooler.balance_cooler(clr_test, ignore_diags=0, store=True, store_name="weight")

    return cooler.Cooler(clr_file)

class TestClass():
    @classmethod
    def setup_class(cls):
        cls.clr = _generate_test_example()

    # create a small toy example inside the test func with a human readable small data
    
    def test_generate_bin_mask(self):
        logger.debug("Start to test generate_bin_mask function...")
        bin_mask = data_util.generate_bin_mask(
            self.clr, [data_util.cis_total_ratio_filter(thres=0.5)], store=True
        )

        # check if bin mask is correctly generated
        assert (bin_mask == np.array([True, False, True, False])).all()
        logger.info("\n\n######### generate_bin_mask Pass the test #########")



    def test_create_filtered_cooler(self, thres=0.5):
        logger.debug("Start to test create_filtered_cooler function...")

        output_path = f"./test_data_util_{thres}filtered.cool"
        bin_mask = data_util.generate_bin_mask(
            self.clr, [data_util.cis_total_ratio_filter(thres=0.5)]
        )
        data_util.create_filtered_cooler(
            output_path, self.clr, bin_mask, chunksize=10_000_000, nproc=16
        )

        logger.debug("Start to examine the result...")
        clr_filtered = cooler.Cooler(output_path)
        coverage_filtered = cooltools.coverage(clr_filtered, ignore_diags=0)
        cis_total_cov_filtered = coverage_filtered[0] / coverage_filtered[1]
        
        # check if all bad bins have nan for cis total ratio in new cool file
        assert np.isnan(cis_total_cov_filtered[~bin_mask]).all()
        # check if all good bins have value for cis total ratio in new cool file
        assert ~np.isnan(cis_total_cov_filtered[bin_mask]).any()
        
        # check if good bins has nan cis-total ratio
        logger.info("\n\n######### create_filtered_cooler Pass the test #########")
