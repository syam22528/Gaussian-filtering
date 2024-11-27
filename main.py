import cv2
import numpy as np
import os

def read_image(file_path):
    """Reads a grayscale image from the given file path."""
    try:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load the image: {file_path}")
        return image
    except Exception as error:
        print(f"Error reading image {file_path}: {error}")
        return None

def apply_gaussian_blur(image):
    """Applies a Gaussian filter to the given grayscale image."""
    gaussian_kernel = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ], dtype=np.float32)
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize kernel to sum to 1

    blurred_image = cv2.filter2D(image, -1, gaussian_kernel)
    return blurred_image

def write_image(output_path, image):
    """Saves the processed image to the specified path."""
    try:
        cv2.imwrite(output_path, image)
    except Exception as error:
        print(f"Error saving image {output_path}: {error}")

def process_folder(input_dir, output_dir):
    """Processes all TIFF images in the input directory and saves results in the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(('.tiff', '.tif')):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_blurred.tiff")

            image = read_image(input_path)
            if image is None:
                print(f"Skipping file {file_name} due to errors.")
                continue

            blurred_image = apply_gaussian_blur(image)
            write_image(output_path, blurred_image)

            print(f"Blurred image saved: {output_path}")

def main():
    input_dir = "input_images"
    output_dir = "output_images"

    process_folder(input_dir, output_dir)

if __name__ == "__main__":
    main()