from PIL import Image, ImageChops
import os

method_type = "raw"
p1 = 90
p2 = 60

def image_minus(path1="", path2=""):
    for layer_id in range(36):
        for steps in range(512, 8192 + 256, 256):
            """
            Subtracts pixel values of two images and saves the result.

            Args:
                path1 (str): Path to the first image.
                path2 (str): Path to the second image.
            """
            img1 = Image.open(f"{path1}/layer_{layer_id}/step_{steps}.png").convert('RGB')
            img2 = Image.open(f"{path2}/layer_{layer_id}/step_{steps}.png").convert('RGB')

            # Subtract pixel values (img1 - img2)
            diff = ImageChops.subtract(img1, img2)

            # Save the resulting image
            savepath = f"figs/{method_type}_diff_{p1}_{p2}/layer_{layer_id}"
            os.makedirs(savepath, exist_ok=True)
            diff.save(f'{savepath}/step_{steps}.png')

if __name__ == "__main__":
    image_minus(f"figs/{method_type}_selected_indices_p{p1}", f"figs/{method_type}_selected_indices_p{p2}")                                                                                                                                                                   