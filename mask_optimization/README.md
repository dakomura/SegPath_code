

- `1.apply_segmentation.py` runs the segmentation models trained on the current dataset and save the results.
- `2.ridge_regression.py` applies ridge regression between the IF-mask intensity and the output probabiities.
- `3.IF_cutoff_optimization.py` optimizes IF intensity cufoff. 
- `4_CELL.MCC_calculation.py` calculates Matthew’s correlation coefficient (MCC) between the prediction and mask for leucocytes, myeloid cells, lymphocytes, plasma cells, and endothelial cells.
- `5_CELL.overlap_ratio_optimization.py` optimizes the nucleus overlap cut-off based on the MCC.
