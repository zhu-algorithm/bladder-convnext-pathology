# Data notice

The metadata CSV files are included for reproducibility auditing. They contain
TCGA/OCELOT identifiers and historical absolute paths from the source compute
environment. No image pixels are included in this repository.

Before running the pipeline, obtain the source images under their applicable
license/terms and either update `image_path` or set `image_root` in the config.
Confirm that the crop size is 224 pixels and that `(x, y)` denotes the upper-left
corner. If the original preprocessing used a different convention, update
`src/data.py` before training.

