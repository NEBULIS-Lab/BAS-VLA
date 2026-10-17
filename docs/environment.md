# Environment setup

The common Python package requirements are listed in
[`requirements.txt`](../requirements.txt). Install the carrier and benchmark
with their own supported dependency set. OpenPI and OpenVLA-OFT may require
separate environments.

To save package versions, repository versions and model file information:

```bash
python scripts/capture_environment.py \
  --repo openpi="$BAS_OPENPI_ROOT" \
  --repo libero="$BAS_LIBERO_ROOT" \
  --artifact adapter=/path/to/residual_adapter.pt \
  --output outputs/environment.json
```

The output lists installed packages, repository versions, and model filenames
and sizes. Additional files can be included with `--artifact LABEL=PATH`.

Provide the trained weights and use the checkpoint's matching normalization,
action layout and feature configuration. See [runtime](runtime.md) for carrier
setup and [training](training.md) for adapter configuration.
