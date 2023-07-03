import webuiapi
import PIL
import os

print(os.getcwd())
import sys
import sdqrcode.Engines.Engine as Engine
from typing import Union


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

    def generate_sd_qrcode(
        self,
        input_image: PIL.Image.Image = None,
        controlnet_input_images: PIL.Image.Image = None,
        return_cn_imgs=False,
    ) -> PIL.Image.Image:
        
        # set the model
        self.api.util_set_model(self.config["global"]["model_name_or_path"])
        print("model set")

        # define controlnet units
        cn_units = []
        for cn_input_img, (name, unit) in zip(
            controlnet_input_images, self.config["controlnet_units"].items()
        ):
            print("unit", unit)
            cn_unit = webuiapi.ControlNetUnit(
                input_image=cn_input_img,
                module="none",
                model=unit["model"],
                pixel_perfect=True,
                weight=unit["weight"],
                guidance_start=unit["start"],
                guidance_end=unit["end"],
            )
            cn_units.append(cn_unit)

        if self.config["global"]["mode"] == "txt2img":
            r = self.api.txt2img(
                seed=self.config["global"]["seed"],
                prompt=self.config["global"]["prompt"],
                negative_prompt=self.config["global"]["negative_prompt"],
                width=self.config["global"]["width"],
                height=self.config["global"]["height"],
                steps=self.config["global"]["steps"],
                sampler_name=self.config["global"]["scheduler_name"],
                cfg_scale=self.config["global"]["cfg_scale"],
                controlnet_units=cn_units,
                batch_size=self.config["global"]["batch_size"],
            )
        if self.config["global"]["mode"] == "img2img":
            r = self.api.img2img(
                seed=self.config["global"]["seed"],
                images=[input_image],
                denoising_strength=self.config["global"]["denoising_strength"],
                prompt=self.config["global"]["prompt"],
                negative_prompt=self.config["global"]["negative_prompt"],
                width=self.config["global"]["width"],
                height=self.config["global"]["height"],
                steps=self.config["global"]["steps"],
                sampler_name=self.config["global"]["scheduler_name"],
                cfg_scale=self.config["global"]["cfg_scale"],
                controlnet_units=cn_units,
                batch_size=self.config["global"]["batch_size"],
            )

        if return_cn_imgs:
            return r.images
        else:
            return r.images[0 : -len(cn_units)]
