import webuiapi
import qrcode
import yaml
import PIL
import os


# Backend enum, one of auto_api, diffusers
class constants:
    AUTO_API = 0
    DIFFUSERS = 1


class Engine:
    def __init__(self, backend_type: constants, config: dict = None):
        """if backend is auto, need to call init_backend_automatic() to initialize the backend"""
        self.config = config
        self.backend = backend_type

    def init_backend_automatic(
        self,
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

    def generate_sd_qrcode(self, qr_img) -> PIL.Image.Image:
        if self.backend == constants.AUTO_API:
            return self.generate_sd_qrcode_auto_api(
                qr_code_img=qr_img,
            )

    def generate_sd_qrcode_auto_api(
        self, qr_code_img, return_cn_imgs=False
    ) -> PIL.Image.Image:
        cn_units = []
        for unit in self.config["controlnet_units"]:
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
            return r.images[0 : -len(cn_units)]  # remove controlnet images

    def generate_sd_qrcode_diffusers(self, config) -> list[PIL.Image.Image]:
        import torch
        from diffusers import StableDiffusionControlNetPipeline

        cn_units = []
        for unit in config["controlnet_units"]:
            cn_units.append(
                ControlNetModel.from_pretrained(
                    unit["model"], torch_dtype=torch.float16
                )
            )

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            config["global"]["model_name_or_path_or_api_name"],
            controlnet=cn_units,
            torch_dtype=torch.float16,
        )
        pipe.enable_model_cpu_offload()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            print("xformers not installed, using default attention")

        ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1e_sd15_tile", torch_dtype=torch.float16
        )


class Sdqrcode:
    def __init__(
        self,
        config_name_or_path: str = "./configs/default.yaml",
        auto_api_hostname: str = "",
        auto_api_port: int = 7860,
        auto_api_https: bool = True,
        auto_api_username: str = "",
        auto_api_password: str = "",
    ):
        """
        Args:
            backend: Backend enum, one of auto_api, diffusers
            model: Model name or path to a pretrained model
            config_name_or_path: Pretrained config name or path if not the same as model_name
            auto_api_hostname: Hostname of the Automatic1111 server
            auto_api_port: Port of the Automatic1111 server (default: 7860)
            auto_api_https: Use HTTPS for the Automatic1111 server
            auto_api_username: Username for the Automatic1111 server (if any)
            auto_api_password: Password for the Automatic1111 server (if any)
        """

        # check if variables are set in env
        auto_api_hostname = (
            os.getenv("AUTO_API_HOSTNAME", "")
            if auto_api_hostname == ""
            else auto_api_hostname
        )
        auto_api_port = (
            os.getenv("AUTO_API_PORT", 7860) if auto_api_port == 7860 else auto_api_port
        )
        auto_api_https = (
            os.getenv("AUTO_API_HTTPS", True)
            if auto_api_https == True
            else auto_api_https
        )
        auto_api_username = (
            os.getenv("AUTO_API_USERNAME", "")
            if auto_api_username == ""
            else auto_api_username
        )
        auto_api_password = (
            os.getenv("AUTO_API_PASSWORD", "")
            if auto_api_password == ""
            else auto_api_password
        )

        # Load backend
        self.backend = (
            constants.AUTO_API if auto_api_hostname != "" else constants.DIFFUSERS
        )

        self.config = get_config(config_name_or_path)

        self.engine = Engine(self.backend, self.config)
        self.engine.init_backend_automatic(
            hostname=auto_api_hostname,
            port=auto_api_port,
            https=auto_api_https,
            username=auto_api_username,
            password=auto_api_password,
        )

        # Load config


def get_config(config_name_or_path: str = "./configs/default.yaml") -> dict:
    default_yaml_path = "./configs/default.yaml"
    custom_yaml_path = config_name_or_path

    with open(default_yaml_path, "r") as f:
        default_config = yaml.safe_load(f)
    with open(custom_yaml_path, "r") as f:
        custom_config = yaml.safe_load(f)

    config = {**default_config, **custom_config}

    return config


def generate_sd_qrcode(
    config_name_or_path: str = "./configs/default.yaml",
    auto_api_hostname: str = "",
    auto_api_port: int = 7860,
    auto_api_https: bool = True,
    auto_api_username: str = "",
    auto_api_password: str = "",
    # TODO: add all the yaml args here
) -> PIL.Image.Image:
    sdqrcode = Sdqrcode(
        config_name_or_path=config_name_or_path,
        auto_api_hostname=auto_api_hostname,
        auto_api_port=auto_api_port,
        auto_api_https=auto_api_https,
        auto_api_username=auto_api_username,
        auto_api_password=auto_api_password,
    )

    error_name_to_enum = {
        "low": qrcode.constants.ERROR_CORRECT_L,
        "medium": qrcode.constants.ERROR_CORRECT_M,
        "quartile": qrcode.constants.ERROR_CORRECT_Q,
        "high": qrcode.constants.ERROR_CORRECT_H,
    }

    config = sdqrcode.config

    qr_img = generate_qrcode_img(
        error_correction=error_name_to_enum[config["qrcode"]["error_correction"]],
        box_size=config["qrcode"]["box_size"],
        border=config["qrcode"]["border"],
        fill_color=config["qrcode"]["fill_color"],
        back_color=config["qrcode"]["back_color"],
        text=config["qrcode"]["text"],
    )
    sd_qr_img = sdqrcode.engine.generate_sd_qrcode(qr_img)

    return sd_qr_img


def generate_qrcode_img(
    error_correction: int = qrcode.constants.ERROR_CORRECT_L,
    box_size: int = 10,
    border: int = 4,
    fill_color: str = "black",
    back_color: str = "white",
    text: str = "https://koll.ai",
) -> PIL.Image.Image:
    qr = qrcode.QRCode(
        version=1,
        error_correction=error_correction,
        box_size=box_size,
        border=border,
    )
    qr.add_data(text)
    qr.make(fit=True)
    qr_img = qr.make_image(
        fill_color=fill_color,
        back_color=back_color,
    )

    # convert to pil
    qr_img = qr_img.convert("RGB")

    return qr_img
