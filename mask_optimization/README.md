# Mask optimization
This subdirectory provides scripts to mask optimization.

## scripts

### `1.apply_segmentation.py` 
This script runs the segmentation models trained on the current dataset and save the results.

usage:
```
python 1.apply_segmentation.py antibody [option] 
```

Input Variable | Description
--- | --- 
--batch | batch size
--rbc | RBC segmentation
--data_dir | input directory
--out_dir | output directory

### `2.ridge_regression.py` 
This script applies ridge regression between the IF-mask intensity and the output probabiities.

usage:
```
python 2.ridge_regression.py antibody 
```

### `3.IF_cutoff_optimization.py` 
This script optimizes IF intensity cufoff. 

usage:
```
python 3.IF_cutoff_optimization.py antibody 
```

### `4_CELL.MCC_calculation.py` 
This script calculates Matthew’s correlation coefficient (MCC) between the prediction and mask for leucocytes, myeloid cells, lymphocytes, plasma cells, and endothelial cells.

usage:
```
python 4_CELL.MCC_calculation.py antibody 
```

### `5_CELL.overlap_ratio_optimization.py` 
This script optimizes the nucleus overlap cut-off based on the MCC.
usage:
```
python 5_CELL.overlap_ratio_optimization.py antibody 
```
