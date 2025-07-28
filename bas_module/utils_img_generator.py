import openai
import json
import time
from pathlib import Path
import io
import base64
import requests
import spacy
import os
import pdb
import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image, make_image_grid

# run 'python -m spacy download en_core_web_sm' to load english language model
nlp = spacy.load("en_core_web_sm")

openai.api_key = os.environ['OPENAI_API_KEY']

class TextPromptGen(object):
    
    def __init__(self):
        super(TextPromptGen, self).__init__()
        self.cnt = 0

    def encode_image_pil(self, image):
        with io.BytesIO() as buffer:
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')


    def generate_prompt(self, image, output_dir):
        # if not os.path.exists(output_dir):
        #     os.mkdir(output_dir)
        api_key = openai.api_key
        base64_image = self.encode_image_pil(image)

        payload = {
            "model": "gpt-4o",
            "messages": [
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": "Please generate a brief scene description as the text input of diffusion models for reproducing a similar scene image. Respond me only with the scene description."
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                ]
            }
            ],
            "max_tokens": 300
        }
        
        prompt = ""

        for i in range(10):
            try:
                response = openai.ChatCompletion.create(model=payload["model"], messages=payload["messages"], timeout=15)
                prompt = response['choices'][0]['message']['content']
                break
            except Exception as e:
                print("Something has been wrong while asking GPT4V. Wait for a second and ask chatGPT4 again!")
                time.sleep(1)
                continue

        openai.api_key = api_key
        # with open('%s/gen.txt'%output_dir, 'w') as f:
        #     f.write("scene description:%s\n"%prompt)
        return prompt
    
    
    def evaluate_image(self, image, output_dir):
        api_key = openai.api_key
        base64_image = self.encode_image_pil(image)
        


        payload = {
            "model": "gpt-4o",
            "messages": [
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": "Could you please tell me whether the generated image has problems like artifacts, blurry, smooth texture, bad quality, distortions, unrealistic and distorted. Your answer should be simply 'Yes' or 'No'."
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                ]
            }
            ],
            "max_tokens": 300
        }
        
        is_good_quality = True
        for i in range(10):
            try:
                response = openai.ChatCompletion.create(model=payload["model"], messages=payload["messages"], timeout=5)
                response = response['choices'][0]['message']['content']
                if response in ['YES', 'Yes', 'yes', 'YES.', 'Yes.', 'yes.']:
                    is_good_quality = False
                elif response in ['NO', 'No', 'no', 'NO.', 'No.', 'no.']:
                    is_good_quality = True
                else:
                    continue
                break

            except Exception as e:
                print("Something has been wrong while asking GPT4V. Wait for a second and ask chatGPT4 again!")
                time.sleep(1)
                continue

        openai.api_key = api_key
        # with open('%s/gen.txt'%output_dir, 'a') as f:
        #     f.write("is_good_quality %02d:%s\n"%(self.cnt, is_good_quality))
        self.cnt = self.cnt + 1
        return is_good_quality
    
    
    

    
    
    
class ImageGen(object):
    def __init__(self, model_path="runwayml/stable-diffusion-inpainting"):
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16"
        )
        self.pipeline.enable_model_cpu_offload()

        self.generator = torch.Generator("cuda").manual_seed(92)
        self.cnt = 0
        
    
    def generate_new_image(self, init_image, mask, prompt, output_dir):

        image = self.pipeline(prompt=prompt, image=init_image, mask_image=mask, generator=self.generator).images[0]
        self.cnt = self.cnt + 1
        
        return image