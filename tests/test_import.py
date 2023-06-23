import unittest


class TestImport(unittest.TestCase):
    def test_import(self):
        # if no error, then pass
        import sys

        import os

        print(os.getcwd())

        sys.path.append("./src/")
        import sdqrcode.sdqrcode as sdqrcode

        print(sdqrcode.generate_qrcode_img.__doc__)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
