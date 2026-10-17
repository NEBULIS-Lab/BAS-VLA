"""CPU checks for intervention construction, independent of policy inference."""
import unittest
import json
from pathlib import Path
import tempfile
import types
import argparse
import numpy as np
from PIL import Image

from bas_vla.integrations.openpi_libero import apply_image_shift, preprocess_image, load_distractor_assets
from bas_vla.preserving.appearance import get_appearance_shift_spec
from bas_vla.preserving.assets import build_distractor_bank
from bas_vla.runtime.scene import SceneVariation, SceneIntervention, apply_layout_shift
from bas_vla.pairs import load_semantic_break_pairs


class AppearanceTests(unittest.TestCase):
    def test_photometric_transform_changes_pixels(self):
        image = np.full((12, 12, 3), 160, dtype=np.uint8)
        result = apply_image_shift(image, get_appearance_shift_spec("bg_dark_dim")["image_shift"], 7)
        self.assertTrue(np.all(result < image))
        expected = int((255 * (160 / 255) ** 1.22 * .64 - 127.5) * .90 + 127.5)
        np.testing.assert_array_equal(result, np.full_like(image, expected))

    def test_full_appearance_spec_cannot_silently_be_used_as_image_config(self):
        with self.assertRaises(ValueError):
            apply_image_shift(np.zeros((8, 8, 3), dtype=np.uint8), get_appearance_shift_spec("margin_noise_band"), 7)

    def test_appearance_preprocessing_uses_resize_then_shift(self):
        image = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(12, 10, 3)
        obs = {"agentview_image": image, "robot0_eye_in_hand_image": image}
        shift = get_appearance_shift_spec("margin_noise_band")["image_shift"]
        base, wrist = preprocess_image(obs, 8, shift, 7, shift_after_resize=True)
        resized = np.asarray(Image.fromarray(image[::-1, ::-1]).resize((8, 8), Image.Resampling.LANCZOS))
        np.testing.assert_array_equal(base, apply_image_shift(resized, shift, 7))
        np.testing.assert_array_equal(wrist, apply_image_shift(resized, shift, 24))

    def test_noise_is_deterministic_and_preserves_interior(self):
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        shift = get_appearance_shift_spec("margin_noise_band_hard")["image_shift"]
        first = apply_image_shift(image, shift, 5)
        np.testing.assert_array_equal(first, apply_image_shift(image, shift, 5))
        np.testing.assert_array_equal(first[20:, 12:88], image[20:, 12:88])
        self.assertFalse(np.array_equal(first[:20], image[:20]))
        self.assertFalse(np.array_equal(first, apply_image_shift(image, shift, 6)))

    def test_clutter_needs_assets_and_preserves_bottom_center(self):
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        assets = [{"name": "red", "image": Image.new("RGBA", (10, 20), (255, 0, 0, 255))}]
        for name in ("margin_object_paste", "perimeter_clutter_strip", "perimeter_clutter_noise_mix"):
            shift = get_appearance_shift_spec(name)["image_shift"]
            with self.assertRaises(ValueError):
                apply_image_shift(image, shift, 7)
            result = apply_image_shift(image, shift, 7, assets)
            self.assertFalse(np.array_equal(result, image))
            np.testing.assert_array_equal(result[30:, 15:85], image[30:, 15:85])

    def test_unknown_operator_rejected(self):
        with self.assertRaises(ValueError):
            apply_image_shift(np.zeros((4, 4, 3), dtype=np.uint8), {"kind": "typo"}, 0)


def fake_env():
    model = types.SimpleNamespace(nq=16, nv=14,
        get_joint_qpos_addr=lambda name: {"robot": 0, "object": (2, 9), "receptacle": (9, 16)}[name],
        camera_name2id=lambda name: 0, cam_pos=np.array([[1., 2., 3.]]),
        cam_quat=np.array([[1., 0., 0., 0.]]))
    objects = {name: types.SimpleNamespace(joints=[name]) for name in ("object", "receptacle")}
    return types.SimpleNamespace(env=types.SimpleNamespace(objects_dict=objects),
        sim=types.SimpleNamespace(model=model, forward=lambda: None))


class SceneTests(unittest.TestCase):
    def test_layout_moves_both_objects_and_keeps_robot_rotations_velocities(self):
        env = fake_env()
        state = np.arange(31, dtype=float)
        shifted = apply_layout_shift(env, state, .03, .02)
        expected = state.copy()
        expected[[3, 10]] += .03
        expected[[4, 11]] += .02
        np.testing.assert_array_equal(shifted, expected)
        np.testing.assert_array_equal(state, np.arange(31, dtype=float))
        for bad in (state[:-1], state.reshape(1, -1)):
            with self.assertRaises(ValueError):
                apply_layout_shift(env, bad, .03, .02)

    def test_camera_jitter_never_accumulates_between_episodes(self):
        env = fake_env()
        operator = SceneIntervention(env, SceneVariation(mode="view_jitter", cam_yaw_deg=90, cam_pitch_deg=0))
        initial = np.arange(31, dtype=float)
        for _ in range(3):
            np.testing.assert_array_equal(operator.apply(env, initial), initial)
            np.testing.assert_allclose(env.sim.model.cam_pos[0], [1.02, 2.05, 3.])
            np.testing.assert_allclose(env.sim.model.cam_quat[0], [np.sqrt(.5), 0, 0, np.sqrt(.5)], atol=1e-14)
        self.assertEqual(operator.metadata()["base_camera_position"], [1., 2., 3.])

    def test_clean_does_not_access_simulator(self):
        operator = SceneIntervention(None, SceneVariation())
        np.testing.assert_array_equal(operator.apply(None, [1., 2.]), [1., 2.])


def fake_joint_env():
    names = {"door_hinge": 0, "drawer_slide": 1, "object_free": 2, "ball": 3}
    model = types.SimpleNamespace(nq=13, nv=11,
        joint_name2id=lambda name: names[name],
        jnt_type=np.array([3, 2, 0, 1]),
        jnt_qposadr=np.array([0, 1, 2, 9]),
        jnt_limited=np.array([True, True, False, False]),
        jnt_range=np.array([[-1., 1.], [0., .4], [0., 0.], [0., 0.]]))
    return types.SimpleNamespace(sim=types.SimpleNamespace(model=model))


class JointStateTests(unittest.TestCase):
    def apply(self, *args):
        from bas_vla.runtime.scene import apply_joint_state
        return apply_joint_state(*args)

    def test_hinge_slide_modify_only_named_qpos_and_leave_source_unchanged(self):
        env = fake_joint_env()
        state = np.arange(25, dtype=float)
        result = self.apply(env, state, {"door_hinge": -.5, "drawer_slide": .3})
        expected = state.copy()
        expected[1:3] = [-.5, .3]
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(state, np.arange(25, dtype=float))

    def test_joint_limits_checked_including_boundary(self):
        env = fake_joint_env()
        state = np.zeros(25)
        self.apply(env, state, {"door_hinge": -1., "drawer_slide": .4})
        for updates in ({"door_hinge": -1.01}, {"drawer_slide": .401}):
            with self.assertRaisesRegex(ValueError, "limit"):
                self.apply(env, state, updates)
        env.sim.model.jnt_limited[0] = False
        self.assertEqual(self.apply(env, state, {"door_hinge": 2.})[1], 2.)

    def test_unknown_free_ball_and_nonscalar_joint_rejected(self):
        env = fake_joint_env()
        for name in ("missing", "object_free", "ball"):
            with self.assertRaises(ValueError):
                self.apply(env, np.zeros(25), {name: 0.})
        env.sim.model.jnt_qposadr[1] = 2
        with self.assertRaisesRegex(ValueError, "scalar"):
            self.apply(env, np.zeros(25), {"door_hinge": 0.})

    def test_nonfinite_nonnumeric_and_invalid_state_rejected(self):
        env = fake_joint_env()
        for value in (float("nan"), float("inf"), True, ".1", [.1]):
            with self.assertRaises(ValueError):
                self.apply(env, np.zeros(25), {"door_hinge": value})
        for state in (np.zeros(24), np.zeros((1, 25)), np.full(25, float("nan"))):
            with self.assertRaises(ValueError):
                self.apply(env, state, {"door_hinge": 0.})

    def test_cli_config_is_applied_and_recorded(self):
        from bas_vla.runtime.scene import add_scene_variation_arguments, scene_variation_from_args
        parser = argparse.ArgumentParser()
        add_scene_variation_arguments(parser)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps({"joint_qpos": {"drawer_slide": .2}}))
            args = parser.parse_args(["--joint-state-config", str(path)])
            operator = SceneIntervention(fake_joint_env(), scene_variation_from_args(args))
            self.assertEqual(operator.apply(fake_joint_env(), np.zeros(25))[2], .2)
            self.assertEqual(operator.metadata()["joint_qpos"], {"drawer_slide": .2})

    def test_config_rejects_duplicate_names_and_missing_explicit_values(self):
        from bas_vla.runtime.scene import load_joint_state_config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            for text in ('{"joint_qpos":{"door_hinge":0,"door_hinge":1}}', '{}',
                         '{"joint_qpos":{}}', '{"joint_qpos":{"door_hinge":NaN}}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_joint_state_config(path)


class CatalogTests(unittest.TestCase):
    def test_semantic_catalog_covers_five_families(self):
        root = Path(__file__).resolve().parents[1]
        pairs = load_semantic_break_pairs(root / "configs/interventions/semantic_pairs.json")
        self.assertEqual({pair.family for pair in pairs}, {"target_object", "container_destination",
            "spatial_relation", "order_constraint", "executability_condition"})
        self.assertEqual(len({pair.pair_id for pair in pairs}), len(pairs))
        for pair in pairs:
            self.assertNotEqual(pair.break_instruction, pair.clean_instruction)
        words = json.loads((root / "configs/interventions/wording_controls.json").read_text())["pairs"]
        self.assertEqual(len(words), 10)
        for pair in words:
            for kind in ("clean", "paraphrase", "format", "redundant"):
                self.assertTrue(pair[kind + "_instruction"])

    def test_asset_builder_crops_exact_source_pixels_and_relocates(self):
        import h5py
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
            with h5py.File(root / "demo.hdf5", "w") as handle:
                handle.create_dataset("data/demo_0/obs/agentview_rgb", data=pixels[None])
            spec = {"assets": [{"name": "crop", "source_hdf5": "demo.hdf5", "demo_key": "demo_0",
                "frame_idx": 0, "crop_box_xyxy": [1, 2, 5, 6]}]}
            output = root / "assets"
            manifest = build_distractor_bank(spec, root, output)
            self.assertEqual(manifest["assets"][0]["asset_path"], "crop.png")
            output.rename(root / "moved")
            assets = load_distractor_assets(root / "moved/manifest.json")
            np.testing.assert_array_equal(np.asarray(assets[0]["image"])[..., :3], pixels[2:6, 1:5])
            with self.assertRaises(FileExistsError):
                build_distractor_bank(spec, root, root / "moved")
            spec["assets"][0]["crop_box_xyxy"] = [0, 0, 100, 100]
            with self.assertRaises(ValueError):
                build_distractor_bank(spec, root, root / "bad")
            self.assertFalse((root / "bad").exists())

    def test_missing_manifest_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            load_distractor_assets(Path("/no-such-intervention-manifest.json"))


if __name__ == "__main__":
    unittest.main()
