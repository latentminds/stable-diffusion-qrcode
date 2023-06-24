import webuiapi
import qrcode
import yaml
import PIL
import os
from pathlib import Path

import sdqrcode.Engines.engine_util as engine_util

CONFIGS = {
    "default_auto": Path(__file__).parent / "configs" / "default_auto.yaml",
    "default_diffusers": Path(__file__).parent / "configs" / "default_diffusers.yaml",
    # Add more configuration files as needed
}


# Backend enum, one of auto_api, diffusers
class constants:
    AUTO_API = 0
    DIFFUSERS = 1


class Sdqrcode:
    def __init__(
        self,
        config_name_or_path_or_dict: str,
        auto_api_hostname: str = None,
        auto_api_port: int = None,
        auto_api_https: bool = None,
        auto_api_username: str = None,
        auto_api_password: str = None,
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

        # Load backend
        self.backend = (
            constants.AUTO_API if auto_api_hostname is None else constants.DIFFUSERS
        )
        if config_name_or_path_or_dict is not dict:
            self.config = get_config(config_name_or_path_or_dict)

        self.engine = engine_util.init_engine(
            hostname=auto_api_hostname,
            port=auto_api_port,
            https=auto_api_https,
            username=auto_api_username,
            password=auto_api_password,
            config=self.config,
        )

    def generate_sd_qrcode(
        self,
        **config_kwargs,
    ) -> PIL.Image.Image:
        self.config = update_config_dict(
            config=self.config,
            **config_kwargs,
        )

        error_name_to_enum = {
            "low": qrcode.constants.ERROR_CORRECT_L,
            "medium": qrcode.constants.ERROR_CORRECT_M,
            "quartile": qrcode.constants.ERROR_CORRECT_Q,
            "high": qrcode.constants.ERROR_CORRECT_H,
        }

        qr_img = generate_qrcode_img(
            error_correction=error_name_to_enum[
                self.config["qrcode"]["error_correction"]
            ],
            box_size=self.config["qrcode"]["box_size"],
            border=self.config["qrcode"]["border"],
            fill_color=self.config["qrcode"]["fill_color"],
            back_color=self.config["qrcode"]["back_color"],
            text=self.config["qrcode"]["text"],
        )

        sd_qr_img = self.engine.generate_sd_qrcode(qr_img)
        return sd_qr_img


def get_config(config_name_or_path: str = "default") -> dict:
    if type(config_name_or_path) == type(dict()):
        return config_name_or_path
    if config_name_or_path in CONFIGS:
        with open(CONFIGS[config_name_or_path], "r") as f:
            config = yaml.safe_load(f)
    else:
        with open(config_name_or_path, "r") as f:
            config = yaml.safe_load(f)

    return config


def update_config_dict(
    config: dict,
    prompt: str = None,
    model_name_or_path_or_api_name: str = None,
    steps: int = None,
    cfg_scale: float = None,
    width: int = None,
    height: int = None,
    seed: int = None,
    controlnet_model_names: list[str] = None,
    controlnet_weights: list[float] = None,
    controlnet_startstops: list[tuple[int, int]] = None,
    qrcode_text: str = None,
    qrcode_error_correction: str = None,
    qrcode_box_size: int = None,
    qrcode_border: int = None,
    qrcode_fill_color: str = None,
    qrcode_back_color: str = None,
):
    if prompt is not None:
        config["global"]["prompt"] = prompt
    if model_name_or_path_or_api_name is not None:
        config["global"][
            "model_name_or_path_or_api_name"
        ] = model_name_or_path_or_api_name
    if steps is not None:
        config["global"]["steps"] = steps
    if cfg_scale is not None:
        config["global"]["cfg_scale"] = cfg_scale
    if width is not None:
        config["global"]["width"] = width
    if height is not None:
        config["global"]["height"] = height
    if seed is not None:
        config["global"]["seed"] = seed

    if controlnet_model_names is not None:
        config["controlnet_units"] = {}
        for i, controlnet_model_name in enumerate(controlnet_model_names):
            config["controlnet_units"][i] = {}
            config["controlnet_units"][i]["model"] = controlnet_model_name
            config["controlnet_units"][i]["weight"] = controlnet_weights[i]
            config["controlnet_units"][i]["start"] = controlnet_startstops[i][0]
            config["controlnet_units"][i]["end"] = controlnet_startstops[i][1]

    if qrcode_text is not None:
        config["qrcode"]["text"] = qrcode_text
    if qrcode_error_correction is not None:
        config["qrcode"]["error_correction"] = qrcode_error_correction
    if qrcode_box_size is not None:
        config["qrcode"]["box_size"] = qrcode_box_size
    if qrcode_border is not None:
        config["qrcode"]["border"] = qrcode_border
    if qrcode_fill_color is not None:
        config["qrcode"]["fill_color"] = qrcode_fill_color
    if qrcode_back_color is not None:
        config["qrcode"]["back_color"] = qrcode_back_color

    return config


def init(
    config: str = "default_diffusers",
    auto_api_hostname: str = None,
    auto_api_port: int = None,
    auto_api_https: bool = None,
    auto_api_username: str = None,
    auto_api_password: str = None,
    **config_kwargs,
):
    """
    config_kwargs:
        model_name_or_path_or_api_name: str = None,
        steps: int = None,
        cfg_scale: float = None,
        width: int = None,
        height: int = None,
        seed: int = None,
        prompt: str = None,
        controlnet_model_names: list[str] = None,
        controlnet_weights: list[float] = None,
        controlnet_startstops: list[tuple[int, int]] = None,
        qrcode_text: str = None,
        qrcode_error_correction: str = None,
        qrcode_box_size: int = None,
        qrcode_border: int = None,
        qrcode_fill_color: str = None,
        qrcode_back_color: str = None,
    """
    # assign variables to config
    config = get_config(config)
    config = update_config_dict(
        config=config,
        **config_kwargs,
    )

    return Sdqrcode(
        config_name_or_path_or_dict=config,
        auto_api_hostname=auto_api_hostname,
        auto_api_port=auto_api_port,
        auto_api_https=auto_api_https,
        auto_api_username=auto_api_username,
        auto_api_password=auto_api_password,
    )


def init_and_generate_sd_qrcode(
    config_name_or_path: str = "default",
    auto_api_hostname: str = "",
    auto_api_port: int = 7860,
    auto_api_https: bool = True,
    auto_api_username: str = "",
    auto_api_password: str = "",
    **config_kwargs,
) -> tuple[PIL.Image.Image, Sdqrcode]:
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
        os.getenv("AUTO_API_HTTPS", True) if auto_api_https == True else auto_api_https
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

    sd_qr_generator = init(
        config=config_name_or_path,
        auto_api_hostname=auto_api_hostname,
        auto_api_port=auto_api_port,
        auto_api_https=auto_api_https,
        auto_api_username=auto_api_username,
        auto_api_password=auto_api_password,
        **config_kwargs,
    )

    sd_qr_img = sd_qr_generator.generate_sd_qrcode()

    return sd_qr_img, sd_qr_generator


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
