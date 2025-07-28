from utils_img_generator import TextPromptGen, ImageGen
from diffusers.utils import load_image, make_image_grid
from torchvision import transforms 
import os
import pdb
from PIL import Image
import torch
import math
import argparse
import glob


PromptGen = TextPromptGen()
ImageGen = ImageGen()


def get_num_of_assets(dataset_dir):
    """Automatically detect the number of mask files in the directory"""
    mask_pattern = os.path.join(dataset_dir, "mask*.png")
    mask_files = glob.glob(mask_pattern)
    # Extract numbers and find the maximum value
    mask_numbers = []
    for mask_file in mask_files:
        filename = os.path.basename(mask_file)
        # Extract the number after mask, e.g., extract 0 from "mask0.png"
        # Only accept files with pattern mask{digit}.png (no underscores or additional characters)
        if filename.startswith("mask") and filename.endswith(".png"):
            try:
                number_str = filename[4:-4]  # Remove "mask" and ".png"
                # Check if the number_str contains only digits (no underscores or other characters)
                if number_str.isdigit():
                    number = int(number_str)
                    mask_numbers.append(number)
            except ValueError:
                continue
    
    if not mask_numbers:
        raise ValueError(f"No mask files (mask*.png) found in directory {dataset_dir}")
    
    # Return the count of consecutive mask files starting from 0
    max_mask_num = max(mask_numbers)
    expected_masks = list(range(max_mask_num + 1))
    
    # Check if all expected mask files exist
    for i in expected_masks:
        mask_path = os.path.join(dataset_dir, f"mask{i}.png")
        if not os.path.exists(mask_path):
            raise ValueError(f"Missing mask file: {mask_path}")
    
    return len(expected_masks)


def main():
    parser = argparse.ArgumentParser(description='Image generation script')
    parser.add_argument('--dataset_dir', type=str, required=True, 
                       help='Dataset directory path')
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    
    # Check if dataset_dir exists
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    # Automatically detect the number of mask files
    num_of_assets = get_num_of_assets(dataset_dir)
    print(f"Detected {num_of_assets} mask files")

    output_dir = dataset_dir + '/gen_img'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in range(int(math.pow(2, num_of_assets))):  # Generate all binary combinations
        binary_name = format(i, '0%db'%num_of_assets)  # Convert number to binary string
        output_dir_sub = output_dir+'/'+binary_name 
        os.makedirs(output_dir_sub, exist_ok=True)  # Create folder named with binary string
        print(f"Created folder: {binary_name}")
        init_image = load_image(dataset_dir+"/img.jpg")
        
        if i == 0:
            continue
        binary_list = [int(char) for char in binary_name]
        
        instance_masks = []
        for j in range(num_of_assets):
            if binary_list[j] == 0:
                continue
            instance_mask_path = os.path.join(dataset_dir, f"mask{j}.png")
            curr_mask = Image.open(instance_mask_path)
            curr_mask = transforms.ToTensor()(curr_mask)[0, None, None, ...]
            instance_masks.append(curr_mask)
        instance_masks = torch.cat(instance_masks)
        instance_masks = torch.max(instance_masks, dim=0)[0].squeeze(0).repeat(3,1,1)
        instance_masks = (~instance_masks.bool()).float()
        instance_masks = transforms.ToPILImage()(instance_masks) 
        
        
        prompt = PromptGen.generate_prompt(init_image, output_dir)
        print(prompt)
        for k in range(10):
            print(f"Generate Image {binary_name} idx{k}")
            image = ImageGen.generate_new_image(init_image, instance_masks, prompt, output_dir_sub)
            image.save('%s/%1d.jpg'%(output_dir_sub,k),'JPEG') 
            
            # while not is_good_quality:
            #     image = ImageGen.generate_new_image(init_image, instance_masks, prompt, output_dir_sub)
                
            #     is_good_quality = PromptGen.evaluate_image(image, output_dir)
            #     print(f"Is good quality {is_good_quality}")
            #     if is_good_quality:
            #         image.save('%s/%1d.jpg'%(output_dir_sub,k),'JPEG')                 


if __name__ == "__main__":
    main()





