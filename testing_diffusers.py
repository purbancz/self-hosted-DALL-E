# from google.colab import output
import huggingface_hub
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

token = open(".huggingface/token", "r")


# CUDA debugging
# print(torch.cuda.is_available())
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.amp)
# print(torch.cuda.amp.autocast)


def counter(filename="runs.dat"):
    with open(filename, "a+") as c:
        c.seek(0)
        val = int(c.read() or 0) + 1
        c.seek(0)
        c.truncate()
        c.write(str(val))
        return str(val)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
  
  
  


# make sure you're logged in with `huggingface-cli login`
use_auth_token=token.read()
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True) 
pipe = pipe.to("cuda")


## standard generated image
prompt = "Evolutionary algorithm"
with torch.autocast("cuda"):
  image = pipe(prompt)["sample"][0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)


image.save(f"outputs/evolutionary_algorithm"+counter()+".png")


## a grid of composed images num_images should equal rows*cols
# num_images = 9
# prompt = ["a photograph of a blue cat looking at a green cat"] * num_images

# with torch.autocast("cuda"):
#   images = pipe(prompt)["sample"]

# grid = image_grid(images, rows=3, cols=3)
# grid.save(f"3x3_bluecat_greencat.png")

