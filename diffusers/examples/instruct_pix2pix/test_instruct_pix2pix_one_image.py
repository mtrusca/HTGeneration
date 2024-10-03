

import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
import PIL
import requests
from tqdm import tqdm
import torch, json
from diffusers import StableDiffusionInstructPix2PixPipeline
import pandas as pd

from PIL import Image, ImageFont
def generate_image(text, img_path, font_path):
    font_size = 36
    font_filepath = font_path
    color = (0, 0, 0)
    font = ImageFont.truetype(font_filepath, size=font_size)
    mask_image = font.getmask(text, "L")
    img = Image.new("RGB", mask_image.size, (255, 255, 255))
    img.im.paste(color, (0, 0) + mask_image.size, mask_image)
    img.save(img_path)

def run(args):
    models1 = [
        'model_pix_265_256'
          ]

    def download_image(url):
       image = PIL.Image.open(url)
       image = image.convert("RGB").resize((256, 256))
       return image

    for i in tqdm(range(len(models1))):
        model_id = models1[i]
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16,
                                                                      local_files_only=True).to("cuda")
        generator = torch.Generator("cuda").manual_seed(0)
        prompt = args.prompt
        img_path = args.img_path
        generate_image(prompt, img_path, args.font_path)
        image = download_image(img_path)
        num_inference_steps = 50
        image_guidance_scale = 1
        guidance_scale = 7.5

        edited_image = pipe(prompt,
         image=image,
         num_inference_steps=num_inference_steps,
         image_guidance_scale=image_guidance_scale,
         guidance_scale=guidance_scale,
         generator=generator,
        ).images[0]

        edited_image.convert("RGB").resize((256, 16)).save(args.img_path_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--img_path_output", type=str)
    parser.add_argument("--font_path", type=str)
    args = parser.parse_args()
    run(args)

