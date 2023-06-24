import unittest
import qrcode
import os
import dotenv
import datetime
import time
from PIL import Image

import sys

sys.path.append("./src/")
from sdqrcode.sdqrcode import Engine, constants, get_config


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
            # path with the date and time and index
            path = f"./tests/imgs_test_results/test_generate_sd_qrcode_auto_api_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{i}.png"

            sd_qr_code_img.save(path)
        self.assertEqual(len(sd_qr_code_imgs), 3)


if __name__ == "__main__":
    unittest.main()
