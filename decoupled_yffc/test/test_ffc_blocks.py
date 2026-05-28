import unittest

import torch

from decoupled_yffc.yffc.models.ffc.ffc_blocks import FFC_BN_ACT


class FFCBlocksTests(unittest.TestCase):
    def test_ffc_bn_act_forward_runs_when_branch_shapes_differ(self):
        module = FFC_BN_ACT(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            ratio_gin=0.5,
            ratio_gout=0.5,
            padding=1,
        )

        x_l = torch.randn(1, 8, 15, 15)
        x_g = torch.randn(1, 8, 16, 16)

        out_l, out_g = module((x_l, x_g))

        self.assertIsInstance(out_l, torch.Tensor)
        self.assertIsInstance(out_g, torch.Tensor)
        self.assertEqual(out_l.shape[-2:], out_g.shape[-2:])


if __name__ == "__main__":
    unittest.main()
