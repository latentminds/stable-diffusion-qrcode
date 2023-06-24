import unittest
import os

import sys

sys.path.append("./src/")
import sdqrcode.sdqrcode as sdqrcode


class TestGetConfig(unittest.TestCase):
    def test_get_config_from_repo(self):
        config = sdqrcode.get_config(config_name_or_path="default")

        self.assertIsNotNone(config)
        self.assertIsInstance(config, dict)
        self.assertIn("global", config)
        self.assertIn("controlnet_units", config)
        self.assertIn("qrcode", config)

    def test_get_config_from_path(self):
        print(os.getcwd())
        print("aaaaaaaaaa")
        config = sdqrcode.get_config(
            config_name_or_path="./tests/test_configs/default_auto.yaml"
        )
        print(config)

        self.assertIsNotNone(config)
        self.assertIsInstance(config, dict)
        self.assertIn("global", config)
        self.assertIn("controlnet_units", config)
        self.assertIn("qrcode", config)


if __name__ == "__main__":
    unittest.main()
