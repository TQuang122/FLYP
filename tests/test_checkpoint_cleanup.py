import os
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for module_name in [
    "clip.clip",
    "clip.loss",
    "src.datasets.common",
    "src.models.eval",
    "src.models.modeling",
    "src.models.utils",
    "src.models.zeroshot",
    "src.datasets.laion",
    "src.datasets",
]:
    sys.modules.setdefault(module_name, types.ModuleType(module_name))

sys.modules["clip.loss"].ClipLoss = object
sys.modules["src.datasets.common"].get_dataloader = object
sys.modules["src.datasets.common"].maybe_dictionarize = object
sys.modules["src.models.eval"].evaluate = object
sys.modules["src.models.modeling"].ClassificationHead = object
sys.modules["src.models.modeling"].CLIPEncoder = object
sys.modules["src.models.modeling"].ImageClassifier = object
sys.modules["src.models.utils"].cosine_lr = object
sys.modules["src.models.utils"].torch_load = object
sys.modules["src.models.utils"].LabelSmoothing = object
sys.modules["src.models.utils"].get_logits = object
sys.modules["src.models.zeroshot"].get_zeroshot_classifier = object
sys.modules["src.datasets.laion"].get_data = object

flyp_loss = importlib.import_module("src.models.flyp_loss")
prepare_checkpoint_dir_for_save = flyp_loss.prepare_checkpoint_dir_for_save


class CheckpointCleanupTests(unittest.TestCase):
    def test_prepare_checkpoint_dir_keeps_space_for_next_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for epoch in range(3):
                for prefix in ("checkpoint", "optim", "scaler"):
                    with open(os.path.join(tmpdir, f"{prefix}_{epoch}.pt"), "w") as file:
                        file.write("data")

            prepare_checkpoint_dir_for_save(tmpdir, keep_last=3)

            remaining = sorted(os.listdir(tmpdir))
            self.assertEqual(
                remaining,
                [
                    "checkpoint_1.pt",
                    "checkpoint_2.pt",
                    "optim_1.pt",
                    "optim_2.pt",
                    "scaler_1.pt",
                    "scaler_2.pt",
                ],
            )

    def test_microbatch_ranges_use_actual_batch_size(self):
        ranges = flyp_loss.make_microbatch_ranges(actual_batch_size=10, microbatch_size=4)

        self.assertEqual(ranges, [(0, 4), (4, 8), (8, 10)])
        self.assertTrue(all(start < end for start, end in ranges))


if __name__ == "__main__":
    unittest.main()
