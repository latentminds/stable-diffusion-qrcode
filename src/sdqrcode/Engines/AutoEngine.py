import Engine
import webuiapi
import PIL


class AutomaticEngine(Engine):
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

        self.api_or_pipeline = webuiapi.WebUIApi(
            hostname, username=username, password=password, port=port, use_https=https
        )

        print("stable diffusion models", self.api_or_pipeline.get_sd_models())
        print(
            "available controlnet models", self.api_or_pipeline.controlnet_model_list()
        )
        print(
            "available controlnet modules",
            self.api_or_pipeline.controlnet_module_list(),
        )

    def generate_sd_qrcode(self, qr_code_img, return_cn_imgs=False) -> PIL.Image.Image:
        cn_units = []
        for name, unit in self.config["controlnet_units"].items():
            cn_unit = webuiapi.ControlNetUnit(
                input_image=qr_code_img,
                module=unit["module"],
                model=unit["model"],
                pixel_perfect=True,
                weight=unit["weight"],
                guidance_start=unit["start"],
                guidance_end=unit["end"],
            )
            cn_units.append(cn_unit)

        r = self.api_or_pipeline.txt2img(
            seed=self.config["global"]["seed"],
            prompt=self.config["global"]["prompt"],
            width=self.config["global"]["width"],
            height=self.config["global"]["height"],
            steps=self.config["global"]["steps"],
            sampler_name=self.config["global"]["sampler_name"],
            cfg_scale=self.config["global"]["cfg_scale"],
            controlnet_units=cn_units,
        )
        if return_cn_imgs:
            return r.images
        else:
            return r.images[0 : -len(cn_units)]
