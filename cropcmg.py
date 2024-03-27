import os
from PIL import Image
import numpy as np
import struct

# This script is designed to crop CMG files, which contain images and associated masks.
# To use this script, follow these steps:

# 1. Set the parameters for the script in main fn:
#    - Specify the path to the CMG file you want to crop (cmg_file_path).
#    - Specify the folder where you want to save the cropped CMG file (output_folder).

# 2. Run the script.

# 3. Check the output folder for the cropped CMG file:
#    - The cropped CMG file will be saved in the specified output folder with the prefix "cropped_" added to the original filename.




CELL_BUFFER = 2  #  pixel buffer for cropping


# Example usage:
# folder_path = "path/to/input_folder"
# output_folder = "path/to/output_folder"
def main():
    # folder with segmented cmg files
    folder_path = r"Z:\FA\Martial\Amy\RawDataSegmentationWithTuner\20227002"
    # folder to save cropped cmgs
    output_folder = r"Z:\FA\Martial\Amy\RawDataSegmentationWithTuner\20227002\Cropped"

    os.makedirs(output_folder, exist_ok=True)

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".cmg"):
            cmg_file_path = os.path.join(folder_path, filename)
            # cropping for each CMG file
            crop_tiles(cmg_file_path, output_folder)
    
    print("Successfully Cropped.")

if __name__ == "__main__":
    main()



def crop_cmg(image, mask):
    # calculating the row-wise and column-wise sums to find non-null regions
    row_sums = np.sum(mask, axis=1)
    col_sums = np.sum(mask, axis=0)

    # determining the bounding box coords
    # argmax fn returns indices of max values along an axis
    top = np.argmax(row_sums != 0) # index of the first non-zero element
    bottom = len(row_sums) - np.argmax(row_sums[::-1] != 0) - 1 # index of the last non-zero element (first non-zero element when row is reversed)
    left = np.argmax(col_sums != 0)
    right = len(col_sums) - np.argmax(col_sums[::-1] != 0) - 1

    # buffering the bounding box coordinates
    top = max(0, top - CELL_BUFFER)
    bottom = min(len(row_sums) - 1, bottom + CELL_BUFFER)
    left = max(0, left - CELL_BUFFER)
    right = min(len(col_sums) - 1, right + CELL_BUFFER)

    # cropping the image and mask
    cropped_image = image[top:bottom+1, left:right+1] # add 1 since 'stop' index is exclusive (not included)
    cropped_mask = mask[top:bottom+1, left:right+1] # we add 1 to make sure the most bottom-right pixel of the cropped img is included

    return cropped_image, cropped_mask


def crop_tiles(cmg_file_path, output_folder):
    folder_path, filename_with_extension = os.path.split(cmg_file_path)
    filename, extension = os.path.splitext(filename_with_extension)

    # read CMG file and extract info
    Images, Masks, Header = readCMG(folder_path, filename)

    # empty lists to hold cropped images and masks
    cropped_images = []
    cropped_masks = []

    # iterate over cells and crop images and masks
    for i, mask_data in enumerate(Masks):
        # crop CMG
        cropped_image, cropped_mask = crop_cmg(Images[i], mask_data)

        # append to the lists
        cropped_images.append(cropped_image)
        cropped_masks.append(cropped_mask)

    updated_header = update_header(Header, cropped_images, cropped_masks)

    # write cropped images and masks to a new CMG file
    writeCMG(updated_header, cropped_images, cropped_masks, output_folder, "cropped_" + filename)



def update_header(Header, cropped_images, cropped_masks):
    num_cropped = len(cropped_images)
    
    # updating the width and height 
    Header[12] = [img.shape[1] for img in cropped_images]  # width
    Header[13] = [img.shape[0] for img in cropped_images]  # height
    
    # updating the positions in the stage axes
    Header[30] = [Header[30][i] - cropped_images[i].shape[0] / 2 for i in range(num_cropped)]  # vorY
    Header[29] = [Header[29][i] - cropped_images[i].shape[1] / 2 for i in range(num_cropped)]  # vorX
    
    return Header







# functions to read and write cmg (paul)


def readCMG(path, filename):
    
#    path=r'\\crcfile11\CI\Paul_Gallagher\MATLAB Applications\Utility Scripts&Functions\CMG_Creator'
#    filename='cancer3'
    slash = '/'
    cmgpath = path + slash + filename + '.cmg'
        
    #with open(cmg_path, "rb") as cmg_file:
    cmgfile = open(cmgpath, "rb")
    cmgdata = cmgfile.read()
    
    btotal = len(cmgdata)
    bcurrent = 0
    icount = 0
    
    
    while (btotal > bcurrent):###Find out how many objects
        NbColorMap = cmgdata[bcurrent+2]
        NbBitMap = cmgdata[bcurrent+96]
        width = int.from_bytes(cmgdata[bcurrent+48 : bcurrent+51], byteorder='little')
        height= int.from_bytes(cmgdata[bcurrent+52 : bcurrent+55], byteorder='little')
        imagesize = width * height
        
        metapadding = 0;
        isMeta = True;
        while(isMeta):
            if(cmgdata[bcurrent+127+metapadding] != 36):
                metapadding = metapadding+1;
            else:
                isMeta = False;
        
        bcurrent = bcurrent + 128 + metapadding
        bcurrent = bcurrent + int(imagesize)*int(NbColorMap)
        bcurrent = bcurrent + int(imagesize)*int(NbBitMap)
        icount = icount + 1
    
    ctotal = icount
    
    Images = []
    Masks = []
    
    Mode = np.zeros(ctotal, dtype='uint8')
    NbColorMap = np.zeros(ctotal, dtype='uint8')
    Class = np.zeros(ctotal, dtype='uint32')
    Screenx = np.zeros(ctotal, dtype='uint32')
    Screeny = np.zeros(ctotal, dtype='uint32')
    Stagex = np.zeros(ctotal, dtype='uint64')
    Stagey = np.zeros(ctotal, dtype='uint64')
    Stagez = np.zeros(ctotal, dtype='uint64')
    Resolution = np.zeros(ctotal, dtype='float')
    LowThreshold = np.zeros(ctotal, dtype='uint16')
    MidThreshold = np.zeros(ctotal, dtype='uint16')
    Group = np.zeros(ctotal, dtype='uint8')
    Width = np.zeros(ctotal, dtype='uint32')
    Height = np.zeros(ctotal, dtype='uint32')
    Accession = np.zeros(ctotal, dtype='uint32')
    Iod = np.zeros(ctotal, dtype='float')
    Fluor = np.zeros(ctotal, dtype='uint8')
    Diagnosis = np.zeros(ctotal, dtype='uint16')
    RedFaction = np.zeros(ctotal, dtype='float')
    GreenFaction = np.zeros(ctotal, dtype='float')
    BlueFaction = np.zeros(ctotal, dtype='float')
    Index = np.zeros(ctotal, dtype='uint32')
    Objective = np.zeros(ctotal, dtype='uint32')
    Calibrated = np.zeros(ctotal, dtype='uint8')
    StackX_int = np.zeros(ctotal, dtype='uint32')
    StackY_int = np.zeros(ctotal, dtype='uint32')
    NbBitMap = np.zeros(ctotal, dtype='uint8')
    CassettePosition = np.zeros(ctotal, dtype='uint8')
    vorx = np.zeros(ctotal, dtype='uint32')
    vory = np.zeros(ctotal, dtype='uint32')
    BestFocusFrame = np.zeros(ctotal, dtype='uint8')
    BackgroundFloat = np.zeros(ctotal, dtype='float')
    PrimaryColourChannel = np.zeros(ctotal, dtype='uint8')
    Layer = np.zeros((ctotal, 2), dtype='uint8')
    Points = np.zeros((ctotal, 9), dtype='uint8')
    NumFeature = np.zeros(ctotal, dtype='uint8')
    RGB_Order = np.zeros((ctotal, 3), dtype='uint8')
    
    bcurrent = 0
    icount = 0
    
    #while (BYTE_TOTAL > BYTE_CURRENT):###
    for n in range(0, ctotal):
    #----------------------------Parse Header Data---------------------------------
        Mode[icount] = cmgdata[bcurrent+1]
        NbColorMap[icount] = cmgdata[bcurrent+2]
        Class[icount] = int.from_bytes(cmgdata[bcurrent+3 : bcurrent+7], byteorder='little')
        Screenx[icount] = int.from_bytes(cmgdata[bcurrent+7 : bcurrent+11], byteorder='little')
        Screeny[icount] = int.from_bytes(cmgdata[bcurrent+11 : bcurrent+15], byteorder='little')
        Stagex[icount] = int.from_bytes(cmgdata[bcurrent+15 : bcurrent+23], byteorder='little')
        Stagey[icount] = int.from_bytes(cmgdata[bcurrent+23 : bcurrent+31], byteorder='little')
        Stagez[icount] = int.from_bytes(cmgdata[bcurrent+31 : bcurrent+39], byteorder='little')
        [Resolution[icount]] = struct.unpack('<f', cmgdata[bcurrent+39 : bcurrent+43])
        LowThreshold[icount] = int.from_bytes(cmgdata[bcurrent+43 : bcurrent+45], byteorder='little')
        MidThreshold[icount] = int.from_bytes(cmgdata[bcurrent+45 : bcurrent+47], byteorder='little')
        Group[icount] = cmgdata[bcurrent+47]
        Width[icount] = int.from_bytes(cmgdata[bcurrent+48 : bcurrent+52], byteorder='little')
        Height[icount]= int.from_bytes(cmgdata[bcurrent+52 : bcurrent+56], byteorder='little')
        Accession[icount]= int.from_bytes(cmgdata[bcurrent+56 : bcurrent+60], byteorder='little')
        [Iod[icount]] = struct.unpack('<f', cmgdata[bcurrent+60 : bcurrent+64])
        Fluor[icount] = cmgdata[bcurrent+64]
        Diagnosis[icount]= int.from_bytes(cmgdata[bcurrent+65 : bcurrent+67], byteorder='little')
        [RedFaction[icount]] = struct.unpack('<f', cmgdata[bcurrent+67 : bcurrent+71])
        [GreenFaction[icount]] = struct.unpack('<f', cmgdata[bcurrent+71 : bcurrent+75])
        [BlueFaction[icount]] = struct.unpack('<f', cmgdata[bcurrent+75 : bcurrent+79])
        Index[icount] = int.from_bytes(cmgdata[bcurrent+79 : bcurrent+83], byteorder='little')
        Objective[icount] = int.from_bytes(cmgdata[bcurrent+83 : bcurrent+87], byteorder='little')
        Calibrated[icount] = cmgdata[bcurrent+87]
        StackX_int[icount] = int.from_bytes(cmgdata[bcurrent+88 : bcurrent+92], byteorder='little')
        StackY_int[icount] = int.from_bytes(cmgdata[bcurrent+92 : bcurrent+96], byteorder='little')
        NbBitMap[icount] = cmgdata[bcurrent+96]
        CassettePosition[icount] = cmgdata[bcurrent+97]
        vorx[icount] = int.from_bytes(cmgdata[bcurrent+98 : bcurrent+102], byteorder='little')
        vory[icount] = int.from_bytes(cmgdata[bcurrent+102 : bcurrent+106], byteorder='little')
        BestFocusFrame[icount] = cmgdata[bcurrent+106]
        [BackgroundFloat[icount]] = struct.unpack('<f', cmgdata[bcurrent+107 : bcurrent+111])
        PrimaryColourChannel[icount] = cmgdata[bcurrent+111]
        for i in range(0, 2):
            Layer[icount, i] = cmgdata[bcurrent+112+i]
        for i in range(0, 9):
            Points[icount, i] = cmgdata[bcurrent+114+i]
        NumFeature[icount] = cmgdata[bcurrent+123]
        for i in range(0, 3):
            RGB_Order[icount, i] = cmgdata[bcurrent+124+i]
            
        imagesize = Width[icount] * Height[icount]
        
        metapadding = 0;
        isMeta = True;
        while(isMeta):
            if(cmgdata[bcurrent+127+metapadding] != 36):
                metapadding = metapadding+1;
            else:
                isMeta = False;
        #print(metapadding)
        bcurrent = bcurrent + 128 + metapadding
    #------------------------------------------------------------------------------
    
    
    #---------------------------Parse Image Data-----------------------------------
        frames = np.zeros((Height[icount], Width[icount], NbColorMap[icount]))
        for n in range(0, NbColorMap[icount]):
            bytevector = np.frombuffer(cmgdata[  bcurrent + int(imagesize)*n  :  bcurrent + int(imagesize)*(n+1) ], dtype=np.uint8)
            frames[:,:,n] = np.reshape(bytevector, (Height[icount], Width[icount]))
            #bytevector.reshape( Height[icount], Width[icount] )
            
        Images.append(frames)
        bcurrent = bcurrent + int(imagesize)*int(NbColorMap[icount])
    
    #------------------------------------------------------------------------------
    
    #--------------------------Parse Mask Data-------------------------------------
        frames = np.zeros((Height[icount], Width[icount], NbBitMap[icount]))
        for n in range(0, NbBitMap[icount]):
            bytevector = np.frombuffer(cmgdata[  bcurrent + int(imagesize)*n  :  bcurrent + int(imagesize)*(n+1) ], dtype=np.uint8)
            frames[:,:,n] = np.reshape(bytevector, (Height[icount], Width[icount]))
            #bytevector.reshape( Height[icount] , Width[icount] )
            
        Masks.append(frames)    
        bcurrent = bcurrent + int(imagesize)*int(NbBitMap[icount])
            
            
    #------------------------------------------------------------------------------
        icount = icount + 1
            
    Header = [Mode, NbColorMap, Class, Screenx, Screeny, Stagex, Stagey, Stagez, Resolution, LowThreshold, MidThreshold, Group, Width, Height, Accession, Iod, Fluor, Diagnosis, RedFaction, GreenFaction, BlueFaction, Index, Objective, Calibrated, StackX_int, StackY_int, NbBitMap, CassettePosition, vorx, vory, BestFocusFrame, BackgroundFloat, PrimaryColourChannel, Layer, Points, NumFeature, RGB_Order]
    
    cmgfile.close()
    
    return [Images, Masks, Header]



def writeCMG(Header, Images, Masks, path, filename):
    numRuns = len(Images)
        
    slash = '/'
    cmgpath = path + slash + filename + '.cmg'
    
    cmgfile = open(cmgpath, "wb")
    
    for n in range(0, numRuns):
        
        H = Images[n].shape[0]
        W = Images[n].shape[1]
        numImage = Images[n].shape[2]
        numMask = Masks[n].shape[2]
        
        #Header[3][n] = Header[28][n] - (W/2);
        #Header[4][n] = Header[29][n] - (H/2);
        
        I_RGB = Images[n]
        I_Bitmap = Masks[n]   
        
        cmgfile.write(b'c')
        cmgfile.write(Header[0][n])#Mode
        cmgfile.write(np.uint8(numImage))#NBColorMap
        cmgfile.write(Header[2][n])#Class
        cmgfile.write(Header[3][n])#Screenx
        cmgfile.write(Header[4][n])#Screeny
        cmgfile.write(Header[5][n])#Stagex
        cmgfile.write(Header[6][n])#Stagey
        cmgfile.write(Header[7][n])#Stagez
        cmgfile.write(bytearray(struct.pack("f",Header[8][n])))#Resolution
        cmgfile.write(Header[9][n])#LowThreshold
        cmgfile.write(Header[10][n])#MidThreshold
        cmgfile.write(Header[11][n])#Group
        cmgfile.write(np.uint32(W))#Width
        cmgfile.write(np.uint32(H))#Height
        cmgfile.write(Header[14][n])#Accession
        cmgfile.write(bytearray(struct.pack("f",Header[15][n])))#Iod
        cmgfile.write(Header[16][n])#Fluor
        cmgfile.write(Header[17][n])#Diagnosis
        cmgfile.write(bytearray(struct.pack("f",Header[18][n])))#RedFraction
        cmgfile.write(bytearray(struct.pack("f",Header[19][n])))#GreenFraction
        cmgfile.write(bytearray(struct.pack("f",Header[20][n])))#BlueFraction
        cmgfile.write(Header[21][n])#Index
        cmgfile.write(Header[22][n])#Objective
        cmgfile.write(Header[23][n])#Calibrated
        cmgfile.write(Header[24][n])#StackX_int
        cmgfile.write(Header[25][n])#StackY_int
        cmgfile.write(np.uint8(numMask))#NbBitMap
        cmgfile.write(Header[27][n])#CassettePosition
        cmgfile.write(Header[28][n])#vorx
        cmgfile.write(Header[29][n])#vory
        cmgfile.write(Header[30][n])#BestFocusFrame
        cmgfile.write(bytearray(struct.pack("f",Header[31][n])))#BackgroundFloat
        cmgfile.write(Header[32][n])#PrimaryColorChannel
        cmgfile.write(Header[33][n][0])#Layer
        cmgfile.write(Header[33][n][1])#Layers
        cmgfile.write(Header[34][n][0])#Points
        cmgfile.write(Header[34][n][1])#Points
        cmgfile.write(Header[34][n][2])#Points
        cmgfile.write(Header[34][n][3])#Points
        cmgfile.write(Header[34][n][4])#Points
        cmgfile.write(Header[34][n][5])#Points
        cmgfile.write(Header[34][n][6])#Points
        cmgfile.write(Header[34][n][7])#Points
        cmgfile.write(Header[34][n][8])#Points
        cmgfile.write(Header[35][n])#NumFeatures
        cmgfile.write(Header[36][n][0])#RGB_Order
        cmgfile.write(Header[36][n][1])#RGB_Order
        cmgfile.write(Header[36][n][2])#RGB_Order
        cmgfile.write(b'$')

        for z in range(0, numImage):
            for y in range(0, H):
                cmgfile.write(bytearray(np.uint8(I_RGB[y, :, z])))
     
        for z in range(0, numMask):
            for y in range(0, H):
                cmgfile.write(bytearray(np.uint8(I_Bitmap[y, :, z])))

    cmgfile.close()
    #return []


# # function to calculate the width and height of an object based on its mask
# def calculate_object_size(mask):
#     if np.sum(mask) == 0:
#         # Handle case where mask is empty
#         return 0, 0
#     height, width, _ = mask.shape
#     return width, height



# def crop_tiles(cmg_file_path, output_folder):
#     folder_path, filename_with_extension = os.path.split(cmg_file_path)
#     filename, extension = os.path.splitext(filename_with_extension)

#     # read CMG file and extract info
#     Images, Masks, Header = readCMG(folder_path, filename)

#     # empty lists to hold cropped images and masks
#     cropped_images = []
#     cropped_masks = []

#     # iterate over cells and calculate bounding boxes
#     for i, mask_data in enumerate(Masks):
#         # get cell center coords (and diameter ??)
#         center_x = Header[29][i]  # vorx
#         center_y = Header[30][i]  # vory
#         image_width = Header[13][i]  # width of image 

#         # object position in the image
#         object_x = Header[3][i]  # ScreenX
#         object_y = Header[4][i]  # ScreenY

#         # calculate the width and height of the object based on its mask
#         width, height = calculate_object_size(mask_data)
        
#         # use its width and height for cropping
#         left = max(0, center_x - width / 2 - CELL_BUFFER)
#         top = max(0, center_y - height / 2 - CELL_BUFFER)
#         right = min(image_width, center_x + width / 2 + CELL_BUFFER)
#         bottom = min(image_width, center_y + height / 2 + CELL_BUFFER)

#         # cast data to int or will run into trouble later
#         top = int(top)
#         bottom = int(bottom)
#         left = int(left)
#         right = int(right)

#         # crop Images and Masks
#         cropped_image = Images[i][top:bottom, left:right]
#         cropped_mask = mask_data[top:bottom, left:right]

#         # append to the lists
#         cropped_images.append(cropped_image)
#         cropped_masks.append(cropped_mask)

#     # write cropped images and masks to a new CMG file
#     writeCMG(Header, cropped_images, cropped_masks, output_folder, "cropped_" + filename)


