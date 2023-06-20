import webuiapi
import qrcode
import yaml
import PIL


# Backend enum, one of auto_api, diffusers
class constants:
    AUTO_API = 0
    DIFFUSERS = 1


class Engine:
    def __init__(self, backend_type: constants, config: dict = None):
        self.config = config
        self.type = backend_type

    def init_engine(
        self,
        auto_modelname: str = "",
        auto_api_hostname: str = "",
        auto_api_port: int = 7860,
        auto_api_https: bool = True,
        auto_api_username: str = "",
        auto_api_password: str = "",
        diffusers_modelname_or_path: str = "",
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

        self.backend = (
            constants.AUTO_API if auto_api_hostname != "" else constants.DIFFUSERS
        )

        if self.backend == constants.AUTO_API:
            return self.init_backend_automatic(
                auto_api_hostname,
                auto_api_port,
                auto_api_https,
                auto_api_username,
                auto_api_password,
            )
        elif self.backend == constants.DIFFUSERS:
            return self.init_backend_diffusers(diffusers_modelname_or_path)
        else:
            raise ValueError(f"Invalid backend: {self.backend}")

    def init_backend_diffusers(self, model_name_or_path_or_api_name: str):
        """
        Args:
            model_name_or_path_or_api_name: Model name or path to a pretrained model
        """
        raise NotImplementedError()

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

    def generate_sd_qrcode(self) -> PIL.Image.Image:
        if self.backend == constants.AUTO_API:
            return self.generate_sd_qrcode_auto_api()

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


class sdqrcode:
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

        # Load backend
        self.backend = (
            constants.AUTO_API if auto_api_hostname != "" else constants.DIFFUSERS
        )
        self.engine = Engine(self.backend)
        self.engine.init_engine(
            auto_api_hostname=auto_api_hostname,
            auto_api_port=auto_api_port,
            auto_api_https=auto_api_https,
            auto_api_username=auto_api_username,
            auto_api_password=auto_api_password,
        )

        # Load config

        self.config = get_config(config_name_or_path)


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
    sdqrcode = sdqrcode(
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

    qr_img = generate_qrcode_img(
        error_correction=error_name_to_enum[
            sdqrcode.config["global"]["error_correction"]
        ],
        box_size=sdqrcode.config["global"]["box_size"],
        border=sdqrcode.config["global"]["border"],
        fill_color=sdqrcode.config["global"]["fill_color"],
        back_color=sdqrcode.config["global"]["back_color"],
        text=sdqrcode.config["global"]["text"],
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
