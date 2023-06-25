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
        sd_qr_code_img, _ = init_and_generate_sd_qrcode(
            config_name_or_path="./src/sdqrcode/config/default_diffusers.yaml"
        )

        self.assertIsNotNone(sd_qr_code_img)

        for i, img in enumerate(sd_qr_code_img):
            img.save(
                f"./tests/imgs_test_results/test_generate_sd_qrcode_{i}_diffusers_txt2img.png"
            )

    def test_generate_sd_qrcode_diffusers_img2img(self):
        sd_qr_code_img, _ = init_and_generate_sd_qrcode(
            config_name_or_path="./src/sdqrcode/configs/img2img_tile_diffusers.yaml"
        )

        self.assertIsNotNone(sd_qr_code_img)

        for i, img in enumerate(sd_qr_code_img):
            img.save(
                f"./tests/imgs_test_results/test_generate_sd_qrcode_{i}_diffusers_img2img.png"
            )


if __name__ == "__main__":
    unittest.main()
