import unittest
import qrcode
import dotenv
import os
import sys

sys.path.append("./src/")
from sdqrcode.sdqrcode import init_and_generate_sd_qrcode


class TestGenerateSDQRCode(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_generate_sd_qrcode_diffusers(self):
        sd_qr_code = init_and_generate_sd_qrcode(
            config_name_or_path="./tests/test_configs/default_diffusers.yaml"
        )

        self.assertIsNotNone(sd_qr_code)

        for i, sd_qr_code_img in enumerate(sd_qr_code):
            sd_qr_code_img.save(
                f"./tests/imgs_test_results/test_generate_sd_qrcode_{i}_diffusers.png"
            )


if __name__ == "__main__":
    unittest.main()
