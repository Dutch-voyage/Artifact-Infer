from PIL import Image, ImageChops
import os

method_type = "raw"

p1 = 90
p2 = 60

def image_add(path=""):
    for steps in range(512, 8192 + 256, 256):
        layer_start = 2
        image = Image.open(f"{path}/layer_{layer_start}/step_{steps}.png").convert('RGB')
        for layer_id in range(layer_start + 1, 36):
            """
            Subtracts pixel values of two images and saves the result.

            Args:
                path1 (str): Path to the first image.
                path2 (str): Path to the second image.
            """
            img1 = Image.open(f"{path}/layer_{layer_id}/step_{steps}.png").convert('RGB')
            
            image = ImageChops.add(image, img1)
        # Save the resulting image
        savepath = f"figs/{method_type}_add_start{layer_start}_end{35}/step_{steps}"
        os.makedirs(savepath, exist_ok=True)
        image.save(f'{savepath}/step_{steps}.png')

if __name__ == "__main__":
    image_add(f"figs/{method_type}_diff_{p1}_{p2}")                                                                                                                                                                   