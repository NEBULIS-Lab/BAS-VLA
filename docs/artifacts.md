# Released artifacts

[Shuaijun/BAS-VLA on Hugging Face](https://huggingface.co/Shuaijun/BAS-VLA) provides a trained OpenVLA residual adapter with its matched LIBERO-10 offline action-calibration inputs. This package covers a selected subset of the paper settings.

## Download

After installing this repository and its dependencies:

```python
from huggingface_hub import snapshot_download

snapshot_download("Shuaijun/BAS-VLA", local_dir="artifacts/bas-vla")
```

The download includes the adapter and vocabulary, action-normalization statistics, matched instruction/action records with demonstration-level splits, source RGB observations, simulator states, BDDL scenes, training configuration, and a CPU loading example. File alignment and source information are described in `data/manifest.json` and `NOTICE.md`.

## Load the adapter

```bash
python artifacts/bas-vla/example.py
```

The example applies the adapter to one supplied held-out record. `--record-index` selects another record. It uses `bas_vla.breaking.residual_adapter.ResidualAdapter` with the bundled vocabulary and metadata.

The checkpoint's carrier is `openvla/openvla-7b`. Its action convention is `openvla_libero10_unnormalized_sign_gripper_offline`: the cached actions use the bundled LIBERO-10 statistics and the gripper transform recorded in `adapter/adapter_metadata.json`. Keep this convention when supplying new actions. OpenPI and OpenVLA-OFT rollout evaluation requires an adapter matched to the selected carrier and the runner's action space.

The adapter checkpoint preserves the training artifact's weights. `training_config.json` describes its objective, and `offline_metrics.json` contains the corresponding offline summaries. The [training guide](training.md) describes how to train adapters from matched records for other carrier settings.

For perturbation construction and task configurations, see [interventions](interventions.md). For live robot evaluation, configure the [hardware backend and calibration](robot.md).
