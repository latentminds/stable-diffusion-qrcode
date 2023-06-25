import webuiapi
import PIL
import os

print(os.getcwd())
import sys
import sdqrcode.Engines.Engine as Engine


class AutomaticEngine(Engine.Engine):
    def __init__(
        self,
        config,
        hostname: str,
        port: int = 7860,
        https: bool = False,
        username: str = "",
        password: str = "",
    ):
        """
        Args:
            hostname: Hostname of the Automatic1111 server
            port: Port of the Automatic1111 server (default: 7860)
            https: Use HTTPS for the Automatic1111 server
            username: Username for the Automatic1111 server (if any)
            password: Password for the Automatic1111 server (if any)
        """
        super().__init__(config)

        port = 443 if https else port

        self.api = webuiapi.WebUIApi(
            hostname, username=username, password=password, port=port, use_https=https
        )

    def generate_sd_qrcode(self, qr_code_img, return_cn_imgs=False) -> PIL.Image.Image:
        cn_units = []
        for name, unit in self.config["controlnet_units"].items():
            print("unit", unit)
            cn_unit = webuiapi.ControlNetUnit(
                input_image=qr_code_img,
                module="none",
                model=unit["model"],
                pixel_perfect=True,
                weight=unit["weight"],
                guidance_start=unit["start"],
                guidance_end=unit["end"],
            )
            cn_units.append(cn_unit)

        r = self.api.txt2img(
            seed=self.config["global"]["seed"],
            prompt=self.config["global"]["prompt"],
            negative_prompt=self.config["global"]["negative_prompt"],
            width=self.config["global"]["width"],
            height=self.config["global"]["height"],
            steps=self.config["global"]["steps"],
            sampler_name=self.config["global"]["sampler_name"],
            cfg_scale=self.config["global"]["cfg_scale"],
            controlnet_units=cn_units,
            batch_size=self.config["global"]["batch_size"],
        )
        if return_cn_imgs:
            return r.images
        else:
            return r.images[0 : -len(cn_units)]
