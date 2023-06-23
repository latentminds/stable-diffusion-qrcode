import unittest
from sdqrcode.sdqrcode import generate_qrcode_img
import qrcode


class TestGenerateQRCodeImg(unittest.TestCase):
    def test_generate_qrcode_img(self):
        error_correction = qrcode.constants.ERROR_CORRECT_Q
        box_size = 20
        border = 2
        fill_color = "red"
        back_color = "yellow"
        text = "https://github.com/github/copilot-preview"

        qr_img = generate_qrcode_img(
            error_correction=error_correction,
            box_size=box_size,
            border=border,
            fill_color=fill_color,
            back_color=back_color,
            text=text,
        )

        self.assertIsNotNone(qr_img)
        qr_img.save("./tests/imgs_test_results/test_qr_code.png")


if __name__ == "__main__":
    unittest.main()
