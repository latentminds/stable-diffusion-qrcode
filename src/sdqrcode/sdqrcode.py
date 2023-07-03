import webuiapi
import qrcode
import yaml
import PIL
import os
import urllib.request
from pathlib import Path
import requests
from io import BytesIO
import sdqrcode.Engines.engine_util as engine_util
from typing import Union

try:
    import torch
except:
    pass

CONFIGS = {
    "default_auto":           Path(__file__).parent / "configs" / "default_auto.yaml",
    "default_diffusers":      Path(__file__).parent / "configs" / "default_diffusers.yaml",
    "brightness_auto":        Path(__file__).parent / "configs" / "brightness_auto.yaml",
    "brightness_diffusers":   Path(__file__).parent / "configs" / "brightness_diffusers.yaml",
    "img2img_tile_auto":      Path(__file__).parent / "configs" / "img2img_tile_auto.yaml",
    "img2img_tile_diffusers": Path(__file__).parent / "configs" / "img2img_tile_diffusers.yaml",
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
        torch_dtype = None,
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
            torch_dtype: (only diffusers) Torch dtype to use for the model (default: torch.float32)
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
            torch_dtype=torch_dtype,
        )

    def generate_sd_qrcode(
        self,
        qr_img: PIL.Image.Image = None,
        input_image: PIL.Image.Image = None, # for img2img
        controlnet_input_images: list[Union[PIL.Image.Image, str]] = None,
        return_cn_imgs: bool = False,
        **config_kwargs,
    ) -> list[PIL.Image.Image]:
        """
        Args:
            qr_img: PIL image of QR code, if None will generate a QR code from config
            return_cn_imgs: Return the controlnet images
        """

        self.config = update_config_dict(
            config=self.config,
            **config_kwargs,
        )
        
        # generate qr code if not provided

        if qr_img is None:
            error_name_to_enum = {
                "low": qrcode.constants.ERROR_CORRECT_L,
                "medium": qrcode.constants.ERROR_CORRECT_M,
                "quartile": qrcode.constants.ERROR_CORRECT_Q,
                "high": qrcode.constants.ERROR_CORRECT_H,
            }

            qr_img = generate_qrcode_img(
                error_correction=error_name_to_enum[self.config["qrcode"]["error_correction"]],
                box_size=self.config["qrcode"]["box_size"],
                border=self.config["qrcode"]["border"],
                fill_color=self.config["qrcode"]["fill_color"],
                back_color=self.config["qrcode"]["back_color"],
                text=self.config["qrcode"]["text"],
            )
        
        # set img2img input image
        input_image = None
        if self.config["global"]["mode"] == "img2img":
            if self.config["global"]["input_image"] == "qrcode":
                input_image = qr_img
            else:
                input_image = read_image(self.config["global"]["input_image"])
            
        
        
        # set controlnet input images
        controlnet_input_images = []
        for _, unit in self.config["controlnet_units"].items():
            if unit["cn_input_image"] == "qrcode":
                cn_input_img = qr_img
            else:
                cn_input_img = read_image(unit["cn_input_image"])
            controlnet_input_images.append(cn_input_img)
            
        sd_qr_imgs = self.engine.generate_sd_qrcode(input_image, controlnet_input_images)
        return sd_qr_imgs


def get_config(config_name_or_path: str = "default_diffusers") -> dict:
    if type(config_name_or_path) == type(dict()):
        return config_name_or_path
    if config_name_or_path in CONFIGS:
        with open(CONFIGS[config_name_or_path], "r") as f:
            config = yaml.safe_load(f)
    else:
        with open(config_name_or_path, "r") as f:
            config = yaml.safe_load(f)

    return config

def read_image(path_or_url):
    try:
        # First, try to open the image assuming the input is a local file path.
        return PIL.Image.open(path_or_url)
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        # If that failed, try to open the PIL.image assuming the input is a URL.
        try:
            response = requests.get(path_or_url)
            response.raise_for_status()  # Raise an exception if the GET request was not successful.
            return PIL.Image.open(BytesIO(response.content))
        except requests.RequestException as e:
            # If the input was not a valid URL or the image could not be downloaded,
            # raise an exception.
            raise ValueError(f"Could not open {path_or_url} as a local file or a URL.") from e

def update_config_dict(
    config: dict,
    mode: str = None,
    prompt: str = None,
    negative_prompt: str = None,
    model_name_or_path: str = None,
    steps: int = None,
    cfg_scale: float = None,
    width: int = None,
    height: int = None,
    seed: int = None,
    batch_size: int = None,
    input_image: Union[PIL.Image.Image, str] = None,
    controlnet_input_images: list[Union[PIL.Image.Image, str]] = None,
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
    if mode is not None:
        config["global"]["mode"] = mode
    if prompt is not None:
        config["global"]["prompt"] = prompt
    if negative_prompt is not None:
        config["global"]["negative_prompt"] = negative_prompt
    if model_name_or_path is not None:
        config["global"]["model_name_or_path"] = model_name_or_path
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
    if batch_size is not None:
        config["global"]["batch_size"] = batch_size
    if input_image is not None:
        config["global"]["input_image"] = input_image
    if controlnet_input_images is not None:
        assert len(controlnet_input_images) == len(config["controlnet_units"].keys()), "Number of controlnet input images must match number of controlnet units"
        for (i, cn_input_image), cn_name in zip(enumerate(controlnet_input_images), config["controlnet_units"].keys()):
            config["controlnet_units"][cn_name]["cn_input_image"] = cn_input_image
    if controlnet_weights is not None:
        assert len(controlnet_weights) == len(config["controlnet_units"].keys()), "Number of controlnet weights must match number of controlnet units"
        for (i, cn_weight), cn_name in zip(enumerate(controlnet_weights), config["controlnet_units"].keys()):
            config["controlnet_units"][cn_name]["weight"] = cn_weight
    if controlnet_startstops is not None:
        assert len(controlnet_startstops) == len(config["controlnet_units"].keys()), "Number of controlnet startstops must match number of controlnet units"
        for (i, cn_startstop), cn_name in zip(enumerate(controlnet_startstops), config["controlnet_units"].keys()):
            config["controlnet_units"][cn_name]["start"] = cn_startstop[0]
            config["controlnet_units"][cn_name]["end"] = cn_startstop[1]
    
    # TODO: add self.update_models
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
    torch_dtype = None,
    **config_kwargs,
):
    """
    config_kwargs:
        model_name_or_path: str = None,
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
        torch_dtype=torch_dtype,
        
    )


def init_and_generate_sd_qrcode(
    config_name_or_path: str = "default_diffusers",
    auto_api_hostname: str = "",
    auto_api_port: int = 7860,
    auto_api_https: bool = True,
    auto_api_username: str = "",
    auto_api_password: str = "",
    torch_dtype = None,
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
        torch_dtype=torch_dtype,
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
