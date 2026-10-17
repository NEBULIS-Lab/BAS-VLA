# Intervention construction

The release includes concrete constructors for all intervention families in the
paper's taxonomy. An intervention specifies what changes; an evaluation manifest
also binds the task, carrier, initial states, seeds, and success predicates.

| Family | Construction | Entry point or configuration |
| --- | --- | --- |
| Illumination | Gamma, brightness, contrast, channel scaling; dark floor/wall scene properties | `bg_dark_dim`, `warm_color_cast` in `bas_vla/preserving/appearance.py` |
| Style | LIBERO floor/wall texture selection with the same object geometry | `style_swap`; OFT `--scene-style style_swap` |
| Clutter | One margin crop or three perimeter crop overlays | `margin_object_paste`, `perimeter_clutter_strip` |
| Noise | Seeded Gaussian noise in the top and side bands | `margin_noise_band`, `margin_noise_band_hard` |
| Clutter + noise | Perimeter crops composed with band noise | `perimeter_clutter_noise_mix` |
| Layout / viewpoint | Shared XY translation of movable objects; named-camera translation and rotation | `bas_vla/runtime/scene.py`; `--scene-variation` |
| Wording control | Curated paraphrases, redundant wording, and formatted instructions | `configs/interventions/wording_controls.json` |
| Target object | Matched clean/control/changed-target instructions and object-scene BDDL assets | `configs/interventions/semantic_pairs.json`, `configs/evaluation/dual_goal_pairs.json`, `bddl/` |
| Destination | Back-to-front caddy compartment instruction change | `break_container_destination` in `semantic_pairs.json` |
| Spatial relation | Between-to-next-to target relation instruction change | `break_spatial_relation` in `semantic_pairs.json` |
| Subgoal order | Microwave placement/closing order swap | `break_order_constraint` in `semantic_pairs.json` |
| Executability | Stove on/off and drawer closed/open instruction changes; explicit hinge/slide joint-state changes | Semantic catalog plus `runtime/scene.py:apply_joint_state`, `--joint-state-config` |

The semantic catalog contains six concrete triplets spanning five semantic
families. It preserves each scene and changes the instruction. The main
object-swap protocols additionally provide scene assets containing the changed
target. The wording catalog contains ten LIBERO-10 tasks, each with clean,
paraphrase, format, and redundant variants.

## Inspect and construct on CPU

List the exact appearance presets and geometric defaults:

```bash
python3 scripts/construct_intervention.py list
```

Export a catalog instruction:

```bash
python3 scripts/construct_intervention.py prompt \
  --pair-id break_container_destination --condition break

python3 scripts/construct_intervention.py prompt \
  --catalog configs/interventions/wording_controls.json \
  --pair-id libero10_05 --condition redundant
```

Apply a noise or image-only appearance operator to an RGB frame:

```bash
python3 scripts/construct_intervention.py image \
  --input frame.png --output frame_noise.png \
  --preset margin_noise_band_hard --seed 7
```

This command expects an already oriented image and writes a new PNG. Presets
containing simulator scene properties use the evaluation environment so the
floor and wall changes are rendered as part of the scene.

## Clutter assets

The crop specification identifies the demonstration, frame, and pixel boxes for
the three distractors. Set `--dataset-root` to the LIBERO dataset directory
containing `libero_10/`:

```bash
python3 -m pip install -e '.[assets]'
python3 scripts/prepare_distractor_assets.py \
  --dataset-root /path/to/LIBERO/datasets \
  --output-dir artifacts/distractor_bank

python3 scripts/construct_intervention.py image \
  --input frame.png --output frame_clutter.png \
  --preset perimeter_clutter_strip --seed 7 \
  --distractor-manifest artifacts/distractor_bank/manifest.json
```

Exported manifest paths are relative to the asset directory. Crops preserve the
source frame orientation. Clutter modifies observed pixels and leaves simulator
objects and collision geometry unchanged. The operators require a nonempty asset
manifest and preserve existing output files.

## Apply in the evaluation runners

`scripts/eval_openpi_appearance_libero.py` accepts all appearance presets through
`--shift-preset`; clutter presets also take `--distractor-manifest`. Its image
path rotates the two raw camera views, resizes them, then applies the selected
appearance transform. The seed is the rollout timestep, with a `+17` offset for
the wrist view. `scripts/eval_openpi_libero.py` retains its interference path:
transform before resizing and use the seed derived from run seed, task, episode,
and timestep. The two paths retain their respective protocol conventions.

Both the OpenPI appearance runner and `scripts/eval_openvla_oft_libero.py` accept:

```text
--scene-variation layout_shift --layout-dx 0.03 --layout-dy 0.02
--scene-variation view_jitter --cam-dx 0.02 --cam-dy 0.05 --cam-dz 0 \
  --cam-yaw-deg 5 --cam-pitch-deg -3 --camera-name agentview
```

Translations use simulator length units and rotations use degrees. Layout
translates object free joints together, retaining object orientation and relative
position; task selection determines whether that translation preserves the goal
and reachability. View jitter changes only the selected camera. Camera offsets
are anchored to its original pose on every episode. The runners record the
selected variation parameters with the episode results.

For the taxonomy's scene-side open/closed state intervention, both runners also
accept `--joint-state-config task_joint_state.json`. The file contains exactly
one top-level key, `joint_qpos`, whose value maps actual scene joint names to
explicit numeric qpos values. Supply those names and values from the task's
scene assets. Hinge values use radians; slide values use model length units.
There are no default joint values or open/closed thresholds.

`apply_joint_state` accepts only scalar hinge and slide joints, checks finite
values and enabled model joint limits, and changes only the named qpos entries
in a copied initial state. Time, other qpos entries, and velocities stay intact.
The chosen values are recorded with scene metadata. A runner invocation applies
the configuration to every episode, including all its instruction variants;
use separate matched original/changed scene configurations when comparing an
executability state change. The task assets bind this generic operator to a
specific open/closed case and its success predicates.

Use the semantic catalog with the OFT triplet runner via
`--pairs-config configs/interventions/semantic_pairs.json --pair-id PAIR_ID`.
Select a carrier and normalization key matched to the chosen suite. That runner
uses native original-task success for clean, control, and break. Changed-task
completion uses the separate New/Old/Other evaluator and explicit old/new goals;
see [evaluation.md](evaluation.md).

## Evaluation assets and diagnostic interventions

Task-level reproduction binds these constructors to the selected task and
initial-state assets. `configs/evaluation/nonobject.template.json` describes the
required destination, spatial-relation, and order evaluation manifest.
`bas_vla/evaluation/nonobject.py` evaluates explicit destination/relation
predicates and strict first-completion subgoal order. Its order tracker rejects
future or simultaneous subgoal completion. Benchmark-specific Plus/PRO
selections and their assets use the manifest interface in
[evaluation.md](evaluation.md).

For preserving-side diagnostics, `bas_vla/analysis/masks.py` constructs seeded
foreground dropout at a requested IoU, explicit missed-target masks, and
area-matched random masks. `configs/evaluation/gate_condition_matrix.json`
defines the crossed semantic/visual conditions. These controls are separate
from the observation intervention that defines the rollout condition; see
[diagnostics.md](diagnostics.md).
