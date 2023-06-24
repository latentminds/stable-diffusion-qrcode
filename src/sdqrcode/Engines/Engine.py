import AutoEngine
import DiffusersEngine


def init_engine(**engine_kwargs):
    if (
        "auto_api_hostname" in engine_kwargs
        and engine_kwargs["auto_api_hostname"] is not None
    ):
        import webuiapi

        return AutoEngine(**engine_kwargs)
    else:
        return DiffusersEngine(**engine_kwargs)


class Engine:
    def __init__(self, config):
        self.config = config
    
    def update_config(self, config):
        

    def init_backend():
        pass

    def generate_sd_qrcode():
        pass


# remove controlnet images
