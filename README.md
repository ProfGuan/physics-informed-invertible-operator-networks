# Geophysics-informed invertible operator networks

## Project description

This repository contains the code accompanying the manuscript
**Geophysics-informed invertible operator networks for solving Bayesian
geophysical inversion problems** (*Computers & Geosciences*, manuscript
CAGEO-D-26-00979).

The editor-facing runnable example is a lightweight verification workflow for
the 1D surface-wave dispersion inversion case. It checks that the public data,
source code, and checkpoint can be loaded and produce the saved reference
posterior summary.

## Repository contents

- `GI_ION_master/GI-INO/1D_inverse/1D_IND/`: retained 1D GI-ION notebook,
  source modules, data, and checkpoint.
- `scripts/verify_1d.py`: one-command 1D verification script.
- `reference/1d/`: reference metrics and posterior summary generated from the
  currently published 1D checkpoint.

## Installation

Python 3.11 is required for the tested environment. Python 3.13 is not
supported by the pinned PyTorch version used in this verification workflow.
Install the required packages:

```bash
pip install -r requirements.txt
```

## Data

The 1D verification uses:

- `GI_ION_master/GI-INO/1D_inverse/1D_IND/data/model1D.npy`
- `GI_ION_master/GI-INO/1D_inverse/1D_IND/data/phase.npy`
- `GI_ION_master/GI-INO/1D_inverse/1D_IND/models/GI-INO_model.pt`

The associated dataset DOI is <https://doi.org/10.5281/zenodo.21122156>.

## Quick verification

From the repository root, run:

```bash
pip install -r requirements.txt
python scripts/verify_1d.py
```

The script selects the first sample in the held-out 1D test split, loads
`GI-INO_model.pt`, generates 512 posterior samples with a fixed random seed,
and compares the posterior mean, posterior standard deviation, and
parameter-space MSE against `reference/1d/reference_metrics.json` and
`reference/1d/reference_summary.npz`.

## Expected output

The command writes:

- `outputs/verify_1d/metrics.json`
- `outputs/verify_1d/posterior_summary.npz`
- `outputs/verify_1d/result.png`

`Verification: PASS` means that the MSE absolute difference, posterior-mean
maximum absolute difference, and posterior-standard-deviation maximum absolute
difference are each less than or equal to the implemented tolerance of `1e-7`.
On the tested CPU environment, the verification completed in approximately
3.41 seconds, excluding package installation. This is a technical corroboration
case, not a full reproduction of every manuscript figure and table.

## Full experiment note

The research notebooks for the 1D and 2D experiments remain in the repository.
This verification workflow intentionally covers only the fully public 1D case
and does not retrain the model.

## Third-party seismic-slice data note

The seismic-slice experiment is associated with third-party material from
`shenghanlin/SeismicFoundationModel`. The original third-party material is not
included in this verification package, and this README does not assert its
redistribution terms.

## License

The source code is released under the MIT License. See `LICENSE`.

## Citation

Please cite the manuscript and the Zenodo Version 2 dataset when using this repository.

## Contact

For questions about the code or data, contact the manuscript authors.
