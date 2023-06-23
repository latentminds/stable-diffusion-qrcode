import unittest
import qrcode
from sdqrcode.sdqrcode import Engine, constants, get_config
import os
import dotenv
from PIL import Image


dotenv.load_dotenv("./tests/.env")


class TestEngineAuto(unittest.TestCase):
    def setUp(self):
        self.engine = Engine(backend_type=constants.AUTO_API, config=get_config())
        self.engine.init_backend_automatic(
            hostname=os.getenv("AUTO_API_HOSTNAME"),
            port=os.getenv("AUTO_API_PORT"),
            https=os.getenv("AUTO_API_HTTPS") == "true",
            username=os.getenv("AUTO_API_USERNAME"),
            password=os.getenv("AUTO_API_PASSWORD"),
        )

    def test_generate_sd_qrcode_auto_api(self):
        qr_code_img = Image.open("./tests/imgs_test_results/test_qr_code.png").resize(
            (512, 512)
        )

        sd_qr_code_imgs = self.engine.generate_sd_qrcode_auto_api(
            qr_code_img, return_cn_imgs=True
        )

        for i, sd_qr_code_img in enumerate(sd_qr_code_imgs):
            sd_qr_code_img.save(
                f"./tests/imgs_test_results/test_sd_qr_code_auto_api_{i}.png"
            )
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
