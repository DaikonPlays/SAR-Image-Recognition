import os
from PIL import Image

def tif_to_png(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    folder_contents = os.listdir(input_folder)
    tif_files = []
    for entry in folder_contents:
        if entry.endswith('.tif'):
            tif_files.append(entry)
    for tif_file in tif_files:
        tif_path = os.path.join(input_folder, tif_file)
        png_file = os.path.splitext(tif_file)[0] + '.png'
        png_path = os.path.join(output_folder, png_file)
        with Image.open(tif_path) as img:
            img.save(png_path, 'PNG')
            print(f"Converted {tif_file} to {png_file}")
tif_folder = 'input directory'
png_folder = 'output directory'
tif_to_png(tif_folder, png_folder)
