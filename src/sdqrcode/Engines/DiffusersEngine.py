import sdqrcode.Engines.Engine as Engine

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
import torch
import transformers
from diffusers import UniPCMultistepScheduler
from diffusers import DPMSolverMultistepScheduler
import PIL


class DiffusersEngine(Engine.Engine):
    def __init__(self, config):
        super().__init__(config)

        controlnet_units = []
        for name, unit in self.config["controlnet_units"].items():
            cn_unit = ControlNetModel.from_pretrained(unit["model"])
            controlnet_units.append(cn_unit)

        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config["global"]["model_name_or_path_or_api_name"],
            controlnet=controlnet_units,
        ).to("cuda")

        # todo setup scheduler

    def generate_sd_qrcode(
        self,
        qr_code_img: PIL.Image.Image,
    ) -> PIL.Image.Image:
        controlnet_weights = [
            unit["weight"] for unit in self.config["controlnet_units"].values()
        ]
        controlnet_startstops = [
            (unit["start"], unit["end"])
            for unit in self.config["controlnet_units"].values()
        ]

        result = self.pipeline(
            prompt=self.config["global"]["prompt"],
            width=self.config["global"]["width"],
            height=self.config["global"]["height"],
            num_inference_steps=self.config["global"]["steps"],
            images=[qr_code_img for _ in range(len(controlnet_weights))],
            controlnet_guidance=controlnet_startstops,
            controlnet_guidance_scale=controlnet_weights,
        )

        return result.images[0]
