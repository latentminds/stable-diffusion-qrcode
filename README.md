# Stable Diffusion QR Code
alpha version

call automatic1111 webui to generate qrcodes, will add a pure diffusers version once [this PR is completed](https://github.com/huggingface/diffusers/pull/3770)

# tldr
**Colab:**  <a target="_blank" href="https://colab.research.google.com/github/koll-ai/stable-difusion-qrcode/blob/master/colabs/demo_sdqrcode.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Motivation
There is multiple methodes availables to generate ai qr code with differents controlnets models and params. Some parameters might works better with some stable diffusion checkpoints and it's a pain to find somethings that works consistanly.
This repo aims to easily try and evaluate differents methods, models, params and share them with a simple config file 

# Exemple
(cherry picked)
![file (4)](https://github.com/koll-ai/stable-difusion-qrcode/assets/22277706/435d4a3c-5eca-498e-a8bd-47d2658e6305)

# Install
```
pip install sdqrcode
```

# Usage
```python
import sdqrcode.sdqrcode as sdqrcode

# generate with default params
sd_qr_code = sdqrcode.generate_sd_qrcode(
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
    weight: 0.35
    start: 0.0
    end: 1.0

  - module: inpaint
    model: control_v11f1e_sd15_tile [a371b31b]
    weight: 0.5
    start: 0.35
    end: 0.70

qrcode:
  text: "https://koll.ai"
  error_correction: high # [low, medium, quart, high]
  box_size: 10
  border: 4
  fill_color: black
  back_color: white
```

# Available configs:
## ./configs/default.yaml
This method seem to be the best for me, I use it with the model realistic_visionV2.
It uses [https://huggingface.co/ioclab/control_v1p_sd15_brightness](Controlnet Brightness)] and [https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile](Controlnet Tile)
Here are my firsts thoughts:
* CN brightness should be left as is
* You can play with CN tile parameters to get an image more or less "grid like"

# Todos
[] fix weird import
[] add more configs
[] allow to set the config without having the file in local path
[] more tests
[] try to install the webui in demo colab
[] add diffusers backend
[] add docs

# Contrib
Please don't hesitate to submit a PR to improve the code or submit a config
