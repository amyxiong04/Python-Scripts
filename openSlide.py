import os
from PIL import Image

# setup
# path to openslide binaries on Windows
OPENSLIDE_PATH = r'C:\Users\axiong\Downloads\openslide-bin-4.0.0.2-windows-x64\openslide-bin-4.0.0.2-windows-x64\bin'

# adding openslide binaries to the DLL search path
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from openslide.deepzoom import DeepZoomGenerator



def save_tiles_as_tiff(deepzoom, level, output_folder, tile_size, overlap):
    """
    Saves tiles of the specified dz level as tiffs

    params:
        deepzoom: deep zoom object
        level: deep zoom level from which to save tiles
        output_folder: output folder for saving tiles
    """
    # make output folder for tiles
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    level_dimensions = deepzoom.level_dimensions[level]

    # save each tile 
    level_tiles_x, level_tiles_y = deepzoom.level_tiles[level]
    for tile_y in range(level_tiles_y):
        for tile_x in range(level_tiles_x):
            tile_address = (tile_x, tile_y)
            try:
                # fetch tile data
                tile_data = deepzoom.get_tile(level, tile_address)

                # calculate top left pixel coords of tile
                x = tile_x * (tile_size - overlap)
                y = tile_y * (tile_size - overlap)

                # save the tile as .tiff
                tile_filename = f"tile_{x}_{y}.tiff"
                tile_path = os.path.join(output_folder, tile_filename)
                tile_data.save(tile_path, format='TIFF')
            except Exception as e:
                print(f"Error")
    print("Tiles saved successfully:)")
    print()



def read_svs_and_generate_deepzoom(file_path, tile_size, overlap, limit_bounds):
    """
    read .svs file, generate ad return a deep zoom object for it

    params:
        file_path: path to the .svs
        tile_size: size of each tile in deep zoom object
        overlap: verlap between neighbouring tiles 
        limit_bounds: whether to limit the bounds of deep zoom image
    """
    slide = openslide.open_slide(file_path)
    deepzoom = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=limit_bounds)
    return deepzoom



def process_svs_file(file_path, tile_size, overlap, limit_bounds):
    try:
        slide = openslide.open_slide(file_path)

        # get some openslide properties of the slide
        properties = {
            'File Format': slide.detect_format(file_path),
            'Dimensions': slide.dimensions,
            'Number of Levels': slide.level_count,
            'Level Dimensions': [slide.level_dimensions[i] for i in range(slide.level_count)],
            'Downsamples': [slide.level_downsamples[i] for i in range(slide.level_count)]
        }

        print(f"Properties of {file_path}:")
        for key, value in properties.items():
            print(f"{key}: {value}")

        # geneerate deepzoom obj
        deepzoom = read_svs_and_generate_deepzoom(file_path, tile_size, overlap, limit_bounds)

        # save tiles at the highest res level as .tiffs
        highest_resolution_level = deepzoom.level_count - 1 # cuz 0 based indexing
        output_folder = file_path.replace('.svs', '_tiles')
        save_tiles_as_tiff(deepzoom, highest_resolution_level, output_folder, tile_size, overlap)

        slide.close()

    except openslide.OpenSlideError as e:
        print(f"error with {file_path}: {e}")



def main():
    # folder with .svs files
    folder_path = r'C:\Users\axiong\Desktop\svsFiles'
    tile_size = 2048
    overlap = 64
    limit_bounds = False

    # process each .svs file in the folder 
    for filename in os.listdir(folder_path):
        if filename.endswith('.svs'):
            file_path = os.path.join(folder_path, filename)
            process_svs_file(file_path, tile_size, overlap, limit_bounds)

if __name__ == "__main__":
    main()



# def read_svs_and_generate_deepzoom(file_path, tile_size=2048, overlap=1, limit_bounds=False):
#     """
#     Read an SVS file, generate a Deep Zoom object, and return it.

#     Parameters:
#         file_path (str): Path to the SVS file.
#         tile_size (int): Size of each tile in the Deep Zoom image. Default is 256.
#         overlap (int): Overlap between adjacent tiles. Default is 1.
#         limit_bounds (bool): Whether to limit the bounds of the Deep Zoom image. Default is False.

#     Returns:
#         DeepZoomGenerator: Deep Zoom object generated from the SVS file.
#     """
#     slide = openslide.open_slide(file_path)
#     deepzoom = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=limit_bounds)

#     # Generate tiles
#     output_folder = f"{os.path.splitext(os.path.basename(file_path))[0]}_tiles"
#     os.makedirs(output_folder, exist_ok=True)
#     for level in range(deepzoom.level_count):
#         level_dimensions = deepzoom.level_dimensions[level]
#         level_tiles = deepzoom.level_tiles[level]
#         for tile_x in range(level_tiles[0]):
#             for tile_y in range(level_tiles[1]):
#                 tile = deepzoom.get_tile(level, (tile_x, tile_y))
#                 tile.save(os.path.join(output_folder, f"tile_level{level}_x{tile_x}_y{tile_y}.tiff"), format="TIFF")

    
#     return deepzoom




# def process_svs_file(file_path):
#     try:
#         slide = openslide.open_slide(file_path)

#         # get some properties of the slide
#         properties = {
#             'File Format': slide.detect_format(file_path),
#             'Dimensions': slide.dimensions,
#             'Number of Levels': slide.level_count,
#             'Level Dimensions': [slide.level_dimensions[i] for i in range(slide.level_count)],
#             'Downsamples': [slide.level_downsamples[i] for i in range(slide.level_count)]
#         }

#         print()
#         print(f"Properties of {file_path}:")
#         for key, value in properties.items():
#             print(f"{key}: {value}")
#         print()


#         # Generate Deep Zoom object
#         deepzoom = read_svs_and_generate_deepzoom(file_path)

#         print("Deep Zoom properties:")
#         print("Number of Deep Zoom levels:", deepzoom.level_count)
#         print("Total number of Deep Zoom tiles:", deepzoom.tile_count)
#         print("List of (tiles_x, tiles_y) tuples for each Deep Zoom level:", deepzoom.level_tiles)
#         print("List of (pixels_x, pixels_y) tuples for each Deep Zoom level:", deepzoom.level_dimensions)


#         # # fetching a tile
#         # level = 14
#         # address = (0, 0)  # Example tile address
#         # tile_image = deepzoom.get_tile(level, address)

#         # # Example of getting tile coordinates
#         # tile_coordinates = deepzoom.get_tile_coordinates(level, address)

#         # # Example of getting tile dimensions
#         # tile_dimensions = deepzoom.get_tile_dimensions(level, address)

#         # print("Tile Coordinates:", tile_coordinates)
#         # print("Tile Dimensions:", tile_dimensions)

#         # # Example of displaying the tile image
#         # tile_image.show()


        
#         # # get highest res level
#         # highest_level = slide.level_count - 1

#         # # get region of highest res level
#         # region = slide.read_region((0, 0), highest_level, slide.level_dimensions[highest_level])

#         #  # convert to a PIL image
#         # image = region.convert('RGB')
#         # image.show()

#         # # convert and save as .tiff
#         # tiff_file_path = os.path.splitext(file_path)[0] + '_highest_resolution.tiff'
#         # image.save(tiff_file_path, format='TIFF')

#         # print(f"TIFF file saved for {file_path}")

#         slide.close()

#     except openslide.OpenSlideError as e:
#         print(f"Error processing {file_path}: {e}")


# # iterate over each svs file in the folder 
# for filename in os.listdir(folder_path):
#     if filename.endswith('.svs'):
#         file_path = os.path.join(folder_path, filename)
#         process_svs_file(file_path)

# print("finished executing.")






