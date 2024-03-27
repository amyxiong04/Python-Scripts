from PIL import Image
import os

def convert_jpg_to_tiff(input_folder, output_folder):
    # create an output folder for tiff files if none
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # iterate through each jpg file in the folder to convert to tiff file
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # open the jpg file
            image = Image.open(os.path.join(input_folder, filename))

            # convert to tiff file
            tiff_filename = filename[:filename.rfind(".")] + ".tiff"
            output_path = os.path.join(output_folder, tiff_filename)
            image.save(output_path, "TIFF")


def main():
    input_folder = r"Z:\FA\Martial\Amy\RawData\20227003"
    output_folder = r"Z:\FA\Martial\Amy\RawDataSegmentationWithTuner\20227003 - blue channel"

    convert_jpg_to_tiff(input_folder, output_folder)
    print("Successfully converted.")

if __name__ == "__main__":
    main()
    