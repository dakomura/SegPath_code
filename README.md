# SegPath generation

This repository provides scripts to generate annotation masks for tissue/cell segmentation using immunofluorescence restaining.

- `1.registration_patch_extraction.py` extracts patches from Whole Slide Images (.ndpi) of tissue microarray sections after rigid and non-rigid registration between H&E-stained and immunofluorescence (IF)-restained sections.
- `2_CELL.run_cellpose.py` runs Cellpose to the extrated patches (for cell segmentation).
- `3_CELL.mask_generation.py` generates the segmentation masks based on the patches from IF-restained sections and the Cellpose output. 
- `3_RBC.mask_generation.py` generates the segmentation masks for red blood cells based on the patches from IF-restained sections. 
- `3_REGION.mask_generation.py ` generates the segmentation masks for tissues based on the patches from IF-restained sections. 
- `4.QC_make_summary.py` calculates blur level and the correlation between DAPI and Hematoxylin signal.
- `5.filter_QC.py` filters out patches based blur level and the correlation between DAPI and Hematoxylin signal.
- `6.train_segmentation_model.py` trains the segmentation models (requires MLFlow).
