import os
from PIL import Image, ImageDraw

def overlay_black_patch(input_folder, output_folder):
    """
    Reads a folder containing images and adds a black patch to the center of 10 images,
    with a size of 1/3 of the original image's dimensions. All images remain in the folder,
    but only 10 are affected.

    Args:
        input_folder (str): Path to the folder containing the original images.
        output_folder (str): Path to the folder where the modified images will be saved.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over subdirectories in the input folder
    for entity in os.listdir(input_folder):
        entity_path = os.path.join(input_folder, entity)
        images_folder = os.path.join(entity_path, "images")

        if os.path.isdir(images_folder):
            print(f"Processing entity: {entity}")

            # Create 'images_attacked' folder for the entity
            images_attacked_folder = os.path.join(entity_path, "images_attacked")
            if not os.path.exists(images_attacked_folder):
                os.makedirs(images_attacked_folder)

            # Iterate over images in the 'images' folder
            images = [f for f in os.listdir(images_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
            images_to_attack = images[:10]  # Only affect the first 10 images

            for filename in images:
                img_path = os.path.join(images_folder, filename)
                img = Image.open(img_path)

                if filename in images_to_attack:
                    # Get image dimensions
                    width, height = img.size

                    # Calculate patch dimensions
                    patch_width = width // 3
                    patch_height = height // 3

                    # Calculate patch coordinates (centered)
                    left = (width - patch_width) // 2
                    top = (height - patch_height) // 2
                    right = left + patch_width
                    bottom = top + patch_height

                    # Draw the black patch
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([left, top, right, bottom], fill="black")

                # Save the image (affected or not) in the 'images_attacked' folder
                output_path = os.path.join(images_attacked_folder, filename)
                img.save(output_path)
                if filename in images_to_attack:
                    print(f"Processed image (attacked): {filename} in entity: {entity}")
                else:
                    print(f"Copied image (unaffected): {filename} in entity: {entity}")

    print("Processing complete.")

# Input and output folder paths
input_folder = "/home/kev/instant-ngp/data/unit_attacked"  # Replace with your input folder path
output_folder = "unit_attacked_processed"  # Replace with your output folder path

# Execute the function
overlay_black_patch(input_folder, output_folder)