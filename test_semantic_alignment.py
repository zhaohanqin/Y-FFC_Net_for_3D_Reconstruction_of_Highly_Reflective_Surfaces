import unittest

import torch

from Y_FFC_unified import EnhancementLayers, EnhancedCompositeLoss


class SemanticAlignmentTests(unittest.TestCase):
    def test_enhancement_layers_accept_gradient_and_input_priors(self):
        module = EnhancementLayers()
        features = (
            torch.rand(1, 256, 8, 8),
            torch.rand(1, 128, 16, 16),
            torch.rand(1, 64, 32, 32),
            torch.rand(1, 32, 64, 64),
        )
        g_map = torch.rand(1, 1, 64, 64)
        original_input = torch.rand(1, 1, 64, 64)

        outputs = module(features, g_map=g_map, original_input=original_input)

        self.assertEqual(len(outputs), 4)
        self.assertEqual(outputs[0].shape, (1, 128, 8, 8))
        self.assertEqual(outputs[1].shape, (1, 64, 16, 16))
        self.assertEqual(outputs[2].shape, (1, 32, 32, 32))
        self.assertEqual(outputs[3].shape, (1, 16, 64, 64))

    def test_enhanced_loss_reports_grad_spec_alignment(self):
        criterion = EnhancedCompositeLoss(use_perceptual=False)
        input_img = torch.rand(1, 1, 32, 32)
        target = torch.rand(1, 1, 32, 32)
        gt_specular = (input_img - target).clamp(0, 1)
        pred_diffuse = torch.rand(1, 1, 32, 32)
        est_b = torch.rand(1, 1, 32, 32)
        pred_grad = torch.rand(1, 1, 32, 32)
        gt_grad = torch.rand(1, 1, 32, 32)

        loss, loss_dict = criterion(
            pred_diffuse,
            target,
            est_b,
            gt_specular,
            pred_grad,
            gt_grad,
            input_img=input_img,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("grad_spec", loss_dict)
        self.assertGreaterEqual(loss_dict["grad_spec"], 0.0)


if __name__ == "__main__":
    unittest.main()
