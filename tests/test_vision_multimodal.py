import unittest
import torch
from utils.vision_helper import extract_image_features, DynamicPatchProcessor

class TestVisionMultimodal(unittest.TestCase):
    def test_extract_image_features_fallback(self):
        # Test zero-dependency fallback for extract_image_features
        features = extract_image_features(image_path_or_url=None, vision_dim=1152, num_patches=16)
        self.assertEqual(features.shape, (16, 1152))
        
        # Test passing invalid path (should trigger fallback cleanly without crashing)
        features_invalid = extract_image_features(image_path_or_url="nonexistent.jpg", vision_dim=1152, num_patches=16)
        self.assertEqual(features_invalid.shape, (16, 1152))

    def test_dynamic_patch_processor(self):
        processor = DynamicPatchProcessor(patch_size=14, min_size=224, max_size=1344)
        
        # Test grid dynamic resolution calculation
        w, h = processor.process_dynamic_resolution(800, 600)
        self.assertEqual(w, 672)
        self.assertEqual(h, 448)
        
        # Test dynamic patch extraction
        image_tensor = torch.randn(3, 224, 224)
        patches = processor.extract_patches(image_tensor)
        
        # 224 / 14 = 16 patches per dimension -> 16 * 16 = 256 patches total
        # patch dimension = 3 * 14 * 14 = 588
        self.assertEqual(patches.shape, (256, 588))

if __name__ == "__main__":
    unittest.main()
