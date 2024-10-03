
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
import PIL
import requests
from tqdm import tqdm
import torch, json
from diffusers import StableDiffusionInstructPix2PixPipeline
import pandas as pd

def run(args):
    models1 = [
        'model_pix_265_256'
          ]
    # f = open(args.path_data)
    # data = json.load(f)
    data = pd.read_csv(args.path_data)

    def download_image(url):
       image = PIL.Image.open(url)
       image = image.convert("RGB").resize((256, 256))
       return image

    for i in tqdm(range(len(models1))):
        model_id = models1[i]
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16,
                                                                      local_files_only=True).to("cuda")
        generator = torch.Generator("cuda").manual_seed(0)

        for img_id in range(len(data)):
            print('text', data.iloc[img_id].edit_prompt)
            if img_id > 10: break
            # img_name = img_id + ".jpg"
            # img_path = os.path.join(args.path_images, img_name)
            img_path = data.iloc[img_id].input_image
            image = download_image(img_path)
            # prompt = "rinse hands"
            # prompt = data[img_id][0]
            prompt = data.iloc[img_id].edit_prompt
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

            path = os.path.join(args.output_dir, os.path.basename(data.iloc[img_id].input_image))
            print(path)
            edited_image.convert("RGB").resize((256, 16)).save(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    run(args)
