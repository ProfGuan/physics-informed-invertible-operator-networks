# 1D verification reference

This directory contains the reference output for the lightweight 1D
surface-wave GI-ION verification workflow.

Files:

- `reference_metrics.json`: selected held-out index, random seed, posterior
  sample count, checkpoint filename, checkpoint SHA256, data SHA256 values,
  posterior summary metrics, package versions, and PASS/FAIL metadata.
- `reference_summary.npz`: posterior samples, posterior mean, posterior
  standard deviation, true model, selected index, seed, and sample count.
- `reference_result.png`: simple visual comparison of the true model and
  posterior mean.

The reference was generated from the currently published checkpoint
`GI_ION_master/GI-INO/1D_inverse/1D_IND/models/GI-INO_model.pt`. The author
should confirm that this is the intended checkpoint before release.
