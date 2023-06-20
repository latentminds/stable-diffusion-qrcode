import webuiapi
import qrcode
import yaml


# Backend enum, one of auto_api, diffusers
class constants:
    AUTO_API = 0
    DIFFUSERS = 1


class Engine:
    def __init__(self, backend_type: constants):
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
            constants.AUTO_API if auto_api_hostname is not None else constants.DIFFUSERS
        )

        self.api_or_pipeline = self.init_engine_from_backend(self.backend)

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

        auto_api = webuiapi.WebUIApi(
            hostname, username=username, password=password, port=port, use_https=https
        )

        print("stable diffusion models", self.auto_api.get_sd_models())
        print("available controlnet models", self.auto_api.controlnet_model_list())
        print("available controlnet modules", self.auto_api.controlnet_module_list())

        return auto_api

    def generate_sd_qrcode(self):
        if self.backend == constants.AUTO_API:
            return self.generate_sd_qrcode_auto_api()

    def generate_sd_qrcode_auto_api(self, qr_code_img):
        cn_units = []
        for unit in self.config["controlnet_units"]:
            cn_unit = webuiapi.ControlNetUnit(
                input_image=qr_code_img,
                module=unit["module"],
                model=unit["model"],
                pixel_perfect=True,
                weight=unit["weight"],
                guidance_start=unit["guidance_start"],
                guidance_end=unit["guidance_end"],
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
            constants.AUTO_API if auto_api_hostname is not "" else constants.DIFFUSERS
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
        default_yaml_path = "./configs/default.yaml"
        custom_yaml_path = config_name_or_path

        with open(default_yaml_path, "r") as f:
            default_config = yaml.load(f)
        with open(custom_yaml_path, "r") as f:
            custom_config = yaml.load(f)

        self.config = {**default_config, **custom_config}

    def generate_qrcode_img(self):
        qr = qrcode.QRCode(
            version=1,
            error_correction=self.config["qr_code"]["error_correction"],
            box_size=self.config["qr_code"]["box_size"],
            border=self.config["qr_code"]["border"],
        )
        qr.add_data(self.config["qr_code"]["text"])
        qr.make(fit=True)
        qr_img = qr.make_image(
            fill_color=self.config["qr_code"]["fill_color"],
            back_color=self.config["qr_code"]["back_color"],
        )

        # convert to pil
        qr_img = qr_img.convert("RGB")

        return qr_img
