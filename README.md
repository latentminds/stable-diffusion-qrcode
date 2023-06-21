# Stable Diffusion QR Code
alpha unstable version

call automatic1111 webui to generate qrcodes


# Install
```
pip install sdqrcode
```

# Usage
```python
import sdqrcode

# generate with default params
sd_qr_code = generate_sd_qrcode(
            config_name_or_path="./configs/default.yaml",
            auto_api_hostname=os.getenv("AUTO_API_HOSTNAME"),
            auto_api_port=os.getenv("AUTO_API_PORT"),
            auto_api_https=os.getenv("AUTO_API_HTTPS") == "true",
            auto_api_username=os.getenv("AUTO_API_USERNAME"),
            auto_api_password=os.getenv("AUTO_API_PASSWORD"),
        )
```

This lib uses a yaml file to describe the qrcode generation process. Exemple:
``` yaml
global:
    prompt: "a beautiful landscape"
    model_name_or_path_or_api_name: "6ce0161689"
    steps: 20
    sampler_name: Euler a
    cfg_scale: 7
    width: 512
    height: 512
    seed: -1

controlnet_units:
    - module: inpaint
      model: control_v1p_sd15_brightness [5f6aa6ed]
      weight: 0.5
      start: 0.1
      end: 0.9

    - module: inpaint
      model: control_v11f1e_sd15_tile [a371b31b]
      weight: 0.5
      start: 0.1
      end: 0.9

qrcode:
    text: "https://koll.ai"
    error_correction: high # [low, medium, quart, high]
    box_size: 10
    border: 4
    fill_color: black
    back_color: white
```





# build from source

```bash
git clone
python3 -m build
