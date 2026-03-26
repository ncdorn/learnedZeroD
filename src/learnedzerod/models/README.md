# Bundled neural network checkpoints

Place **junction** and **vessel** trial-0 models here for each `VMR_*` set you ship:

```
{VMR_*}/stenosis_off_symmetric_gen_loss/bifurcations_EL_trial_0/rri_{VMR_*}_pred_{0,1,2}_model
{VMR_*}/stenosis_off_symmetric_gen_loss/bifurcations_EL_vessel_trial_0/rri_{VMR_*}_vessel_pred_{0,1,2}_model
```

From a `learn_lpns` checkout that has trained weights:

```bash
python scripts/sync_bundled_models.py --learn-lpns-root /path/to/learn_lpns
```

Large binary artifacts are often gitignored; keep them on shared storage or release tarballs as needed.
