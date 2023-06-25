
# Stable Diffusion QR Code
alpha version, expect breaking changes

call diffusers pipeline or [Automatic1111 webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) api to generate qrcodes, will add a pure diffusers version once [this PR is completed](https://github.com/huggingface/diffusers/pull/3770)

# tldr

```python
import sdqrcode
sd_qr_images, generator = sdqrcode.init_and_generate_sd_qrcode(config="default_diffusers")
```

**Diffusers Colab:**  <a target="_blank" href="https://colab.research.google.com/github/koll-ai/stable-diffusion-qrcode/blob/master/colabs/demo_sdqrcode_diffusers.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**Automatic1111 Colab:**  <a target="_blank" href="https://colab.research.google.com/github/koll-ai/stable-difusion-qrcode/blob/master/colabs/demo_sdqrcode.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Updates
**June 25:** The diffusers version has been added to the package
**June 23:** a colab with a pure diffusers version without automatic1111 dependencie is now available. It will be added to the package soon

# Motivation
There is multiple methodes availables to generate ai qr code with differents controlnets models and params. Some parameters might works better with some stable diffusion checkpoints and it's a pain to find somethings that works consistanly.
This repo aims to easily try and evaluate differents methods, models, params and share them with a simple config file 

# Exemple
(cherry picked, will add more results later)
![Dalmatian qrcode](https://github.com/koll-ai/stable-difusion-qrcode/assets/22277706/a33a7ae9-3842-4290-b5b2-0104f5339323)

![Swimming pool girl qrcode](https://github.com/koll-ai/stable-difusion-qrcode/assets/22277706/435d4a3c-5eca-498e-a8bd-47d2658e6305)

# Install
```
pip install sdqrcode
pip install git+https://github.com/holwech/diffusers # for controlnet start/stop, I'll update once the PR is merged
```
# Usage Diffusers

```python
import sdqrcode
# init with a default config
generator = sdqrcode.init(config = "default_diffusers")

# or with a custom config
generator = sdqrcode.init(config = "/path/to/config.yaml")

# or you can also set custom config params (base model, controlnet models, steps, ...)
generator = sdqrcode.init(config = "default_diffusers", model_name_or_path="Lykon/DreamShaper")


# Then you can generate according to the config
images = generator.generate_sd_qrcode()

# or with some custom parameters (you can't set the models at this stage)
images = generator.generate_sd_qrcode(
    prompt = "A beautiful minecraft landscape,
    steps = 30,
    cfg_scale = 7 ,
    width = 768,
    height = 768,
    seed = -1,
    controlnet_weights = [0.35, 0.65], # [weight_cn_1, weight_cn_2, ...]
    controlnet_startstops = [(0,1), (0.35, 0.7)], # [(start_cn_1, end_cn_1), ... ]. (0.35, 0.7) means apply CN after 35% of total steps until 70% of total steps 
    qrcode_text = "https://koll.ai" ,
    qrcode_error_correction = "high",
    qrcode_box_size = 10,
    qrcode_border = 4,
    qrcode_fill_color = "black",
    qrcode_back_color = "white",
)
```


# Usage Automatic1111
```python
import sdqrcode

# Use an auto config and define the auto_* params in init to use Automatic1111 backend
generator = sdqrcode.init(
            config_name_or_path = "default_auto",
            auto_api_hostname = "auto_hostname",
            auto_api_port=7860,
            auto_api_https = True,
            auto_api_username = "auto_user",
            auto_api_password = "auto_pass"
        )

# Then you can generate like the diffusers version
images = generator.generate_sd_qrcode()
```
# Config File

This lib uses a yaml file to describe the qrcode generation process. Exemple:
``` yaml
global:
  prompt: "a beautiful minecraft landscape, lights and shadows"
  model_name_or_path: "SG161222/Realistic_Vision_V2.0"
  steps: 20
  # sampler_name: Euler a not implemented yet
  cfg_scale: 7
  width: 768
  height: 768
  seed: -1

controlnet_units:
  brightness:
    model: ioclab/control_v1p_sd15_brightness
    #module: none not implemented yet
    weight: 0.35
    start: 0.0
    end: 1.0

  tile:
    model: lllyasviel/control_v11f1e_sd15_tile
    #module: none not implemented yet
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
## default
This method seem to be the best for me, I use it with the model [realistic_visionV2](https://civitai.com/models/4201/realistic-vision-v20).
It uses [Controlnet Brightness](https://huggingface.co/ioclab/control_v1p_sd15_brightness) and [Controlnet Tile](https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile)
Here are my firsts thoughts:
* CN brightness should be left as is
* You can play with CN tile parameters to get an image more or less "grid like"

# Todos
- [ ] allow to set the model in the config
- [ ] add more configs
- [x] allow to set the config without having the file in local path
- [ ] more tests
- [ ] try to install the webui in demo colab
- [x] add diffusers backend
- [ ] add docs

# Contrib
Please don't hesitate to submit a PR to improve the code or submit a config
