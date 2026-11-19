"""Release regressions that do not require model weights or a GPU."""
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch

from data import ImageDataset
from main import Pivot, build_configs, parse_args


class ImageInputTests(unittest.TestCase):
    def test_unique_discovery_and_rgb_conversion(self):
        with tempfile.TemporaryDirectory() as root:
            Image.new('RGB', (20, 20)).save(Path(root) / 'a.PNG')
            Image.new('RGBA', (20, 20)).save(Path(root) / 'b.png')
            Image.new('L', (20, 20)).save(Path(root) / 'c.jpg')
            dataset = ImageDataset(root=root, resolution=16, device='cpu')
            self.assertEqual(len(dataset), 3)
            for image in dataset:
                self.assertEqual(tuple(image.shape), (3, 16, 16))

    def test_empty_root_and_empty_slice_have_actionable_error(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, '--data_root'):
                ImageDataset(root=root, device='cpu')
            Image.new('RGB', (20, 20)).save(Path(root) / 'a.png')
            with self.assertRaisesRegex(ValueError, '--data_root'):
                ImageDataset(root=root, start_id=1, end_id=1, device='cpu')


class RecipeTests(unittest.TestCase):
    def test_all_published_task_commands_resolve(self):
        recipes = Path(__file__).resolve().parents[1] / 'configs_pivot_cli.txt'
        cells = {}
        for line in recipes.read_text().splitlines():
            cmd = shlex.split(line, comments=True)
            if not cmd:
                continue
            with self.subTest(command=line), patch.object(sys, 'argv', cmd[1:]):
                self.assertEqual(cmd[:2], ['python', 'main.py'])
                args = parse_args()
                data, model, operator, scheduler = build_configs(args)
                prior = model['name']
                dataset = args.data.removeprefix('test-')
                self.assertEqual(args.model, f'{dataset}256{prior}')
                self.assertEqual((data['start_id'], data['end_id']), (0, 100))
                self.assertEqual((args.batch_size, args.seed), (1, 42))
                self.assertEqual(args.num_runs, 4 if args.task == 'phase_retrieval' else 1)
                self.assertEqual((operator['sigma'], args.sigma_y), (0.05, 0.05))
                self.assertEqual(scheduler['num_steps'], args.K)
                self.assertEqual(scheduler['sigma_max'], 10 if prior == 'ldm' else 80)
                key = dataset, prior, args.task
                self.assertNotIn(key, cells)
                cells[key] = (args.K, args.J)

        tasks = {'down_sampling', 'inpainting', 'inpainting_rand', 'gaussian_blur',
                 'motion_blur', 'phase_retrieval', 'nonlinear_blur', 'hdr'}
        self.assertEqual(set(cells), {(dataset, prior, task) for dataset in ('ffhq', 'imagenet')
                                     for prior in ('ddpm', 'ldm') for task in tasks})
        for prior in ('ddpm', 'ldm'):
            for task in tasks:
                self.assertEqual(cells['ffhq', prior, task], cells['imagenet', prior, task])


class VerificationTests(unittest.TestCase):
    def test_commit_improvement_and_keep_anchor_on_tie_or_increase(self):
        sampler = Pivot({'name': 'edm', 'num_steps': 3, 'sigma_max': 1,
                         'sigma_min': 0.1, 'timestep': 'poly-7'}, latent=False, J=2, sigma_y=1)
        anchor = torch.zeros(3, 1, 1, 1)
        sampler.operator = lambda z: 3 * z
        sampler.y = torch.ones_like(anchor)
        # At beta=7 the frozen objective is ((3z-1)^2 + 7z^2)/2 per image.
        # The three curvatures propose 3/16 (minimum), 3/8 (tie), and 0.4 (increase).
        curvature = torch.tensor([9, 1, 0.5], dtype=torch.float64)
        verified, _ = sampler._level(anchor, beta=7, curvature=curvature)
        torch.testing.assert_close(verified.flatten(), torch.tensor([3/16, 0, 0]), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
