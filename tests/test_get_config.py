import unittest

import sys

sys.path.append("./src/")
import sdqrcode.sdqrcode as sdqrcode


class TestGetConfig(unittest.TestCase):
    def test_get_config(self):
        config = sdqrcode.get_config(config_name_or_path="./configs/test.yaml")

        self.assertIsNotNone(config)
        self.assertIsInstance(config, dict)
        self.assertIn("global", config)
        self.assertIn("controlnet_units", config)
        self.assertIn("qrcode", config)


if __name__ == "__main__":
    unittest.main()
