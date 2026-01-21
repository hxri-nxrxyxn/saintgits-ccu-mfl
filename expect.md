# Expected Files and Directories

This repository excludes large data files, dependencies, and build artifacts to maintain a clean structure. Below is a list of directories and files that are expected to exist for the project to function correctly, but are not tracked by git.

## Directories

*   **`data/` (and subdirectories like `img/`, `text/`, `raw_images/`)**: Should contain the raw and processed dataset files.
    *   `airquality.csv`: Raw CSV data.
    *   `raw_images/`: Directory containing raw image files (e.g., `mock_img_0.jpg`).
    *   `img/` & `text/`: Processed feature pickles (`.pkl`) if generating locally.
*   **`fed_multimodal_restcol/preprocess/output/`**: Directory where the data partition script outputs the partitioned data (`partition/`) and extracted features (`feature/`).
*   **`mock_storage/`**: Used by the `RestColClient` in offline/mock mode to store "uploaded" data and model checkpoints.

## Files

*   **`.env`**: Configuration file for environment variables (e.g., `RESTCOL_AUTH_TOKEN`, `STORAGE_ACCESS_KEY`).
*   **`*.pkl`**: Pickle files containing serialized data or model states. These are generated during the data prep and training phases.
*   **`*.so`**: Shared object files (C extensions) used by some Python libraries.

## Setup

To regenerate the missing data and artifacts, follow the setup instructions in `README.md` or run the provided pipeline scripts (e.g., `start_pipeline.sh`).
