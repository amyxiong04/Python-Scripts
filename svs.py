import logging
import numpy as np
from typing import Union, Literal

from pathlib import Path
from ...core import fileio
from ..pipelines import sunet_functions as proc

from PIL import Image
import skimage.io as skio

import tensorflow as tf
from keras.models import Model
from keras.layers import LeakyReLU, Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam

from keras import backend as K
import importlib.util as importutil

if not (importutil.find_spec('tensorflow') is None):
    __framework__ = 'tensorflow'
    framework_name = 'framework_tf'
elif not (importutil.find_spec('torch') is None):
    __framework__ = 'pytorch'
    framework_name = 'framework_torch'
else:
    raise ModuleNotFoundError(("coffi requires one of tensorflow and pytorch "
                               "to be installed."))
__version__ = "3.0.0.0"
current_dir = Path(__file__).parent

K.set_image_data_format('channels_last')

histology = current_dir / f"Weights/histology"
cytology = current_dir / f"Weights/cytology"

class SequentialUNetPipeline:
    @proc.func_timer
    def __init__(
        self, 
        load_by_name: tuple[Literal['histology', 'cytology'], str] = None,
        centers_weights: str = "nuclei_detection_weights.h5", 
        contours_weights: str = "Segmentation-BLUE_model_weights.h5",
        build_model_at_init: bool = True,
        compile_at_init: bool = True
    ) -> None:
        self.centers_model = None
        self.contours_model = None
        # >>> Build the model if build_architecture_at_init is True <<< #
        if build_model_at_init:
            self.centers_model = self.create_unet_centers_model()
            self.contours_model = self.create_unet_contours_model()

        # >>> Compile models if compile_at_init is True <<< #
        if compile_at_init:
            self.centers_model.compile(
                optimizer = Adam(learning_rate = 1e-4), 
                loss = 'binary_crossentropy', 
                metrics = ['accuracy']
            )
            self.contours_model.compile(
                optimizer = Adam(learning_rate = 1e-4), 
                loss = 'binary_crossentropy', 
                metrics = ['accuracy']
            )

            # >>> Load the weights <<< #
            self.weights_loading(load_by_name, 
                                 centers_weights, 
                                 contours_weights)
            
        # >>> Check that the self object is ready to run <<< #
        try:
            self.centers_model.summary(print_fn = logging.info)
        except ValueError:
            logging.error(("Center detection model is not built. Build the "
                        "model before running instance segmentation."))
            
        try:
            self.contours_model.summary(print_fn = logging.info)
        except ValueError:
            logging.error(("Segmentation model is not built. Build the "
                        "model before running instance segmentation."))

    def weights_loading(
        self,
        autoload_weights_by_name: str, 
        centers_weights: str,
        contours_weights: str
    ) -> None:
        # >>> Load the weights <<< #
        if not (autoload_weights_by_name is None):
            mode, name = autoload_weights_by_name
            if mode == 'histology':
                weights_path = f'{histology}/{name}'
            elif mode == 'cytology':
                weights_path = f'{cytology}/{name}'
            else:
                raise FileNotFoundError("Mode doesn't exist.")
            self.centers_model.load_weights(
                f'{weights_path}/Segmentation-RED_model_weights.h5'
            )
            self.contours_model.load_weights(
                f'{weights_path}/Segmentation-BLUE_model_weights.h5'
            )
        if not (centers_weights is None):
            self.centers_model.load_weights(centers_weights)
        if not (contours_weights is None):
            self.contours_model.load_weights(contours_weights)
        
        return None
    

    @proc.func_timer
    def create_unet_centers_model(
        self, 
        input_X: int = 128, 
        input_Y: int = 128,
        input_C: int = 1,
        num_features: int = 32,
        activation_fx: str = LeakyReLU(0.1),
        padding_method: str = 'same',
        kernel_init: str = 'random_uniform',
        with_bias: str = False
    ) -> tf.keras.Model:
        common_params = {
            'activation': activation_fx,
            'padding': padding_method,
            'kernel_initializer': kernel_init,
            'use_bias': with_bias
        }

        inputs = Input((input_X, input_Y, input_C))
    
        conv1_ = Conv2D(num_features, 
                        kernel_size = (5, 5), 
                        input_shape=(input_X, input_Y, input_C), 
                        **common_params)(inputs)
        conv1 = Conv2D(num_features, (5, 5), **common_params)(conv1_)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2_ = Conv2D(2 * num_features, (3, 3), **common_params)(pool1)
        conv2 = Conv2D(2 * num_features, (3, 3), **common_params)(conv2_)
        pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)
        
        conv3_ = Conv2D(4 * num_features, (3, 3), **common_params)(pool2)
        conv3 = Conv2D(4 * num_features, (3, 3 ), **common_params)(conv3_)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4_ = Conv2D(8 * num_features, (3, 3), **common_params)(pool3)
        conv4 = Conv2D(8 * num_features, (3, 3), **common_params)(conv4_)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
        conv5_ = Conv2D(16 * num_features, (3, 3), **common_params)(pool4)
        conv5 = Conv2D(16 * num_features, (3, 3), **common_params)(conv5_)
        
        up6 = Conv2DTranspose(16 * num_features, 
                              kernel_size = (2, 2), 
                              strides = (2, 2), 
                              **common_params)(conv5)
        up6 = concatenate([up6, conv4], axis=3)
        conv6_ = Conv2D(8 * num_features, (3, 3), **common_params)(up6)
        conv6 = Conv2D(8 * num_features, (3, 3), **common_params)(conv6_)
        
        up7 = Conv2DTranspose(8 * num_features, 
                              kernel_size = (2, 2), 
                              strides = (2, 2), 
                              **common_params)(conv6)
        up7 = concatenate([up7, conv3], axis=3)
        conv7_ = Conv2D(4 * num_features, (3, 3), **common_params)(up7)
        conv7 = Conv2D(4 * num_features, (3, 3), **common_params)(conv7_)
        
        up8 = Conv2DTranspose(4 * num_features, 
                              kernel_size = (2, 2), 
                              strides = (2, 2), 
                              **common_params)(conv7)
        up8 = concatenate([up8, conv2], axis=3)
        conv8_ = Conv2D(2*num_features, (3, 3), **common_params)(up8)
        conv8 = Conv2D(2*num_features, (3, 3), **common_params)(conv8_)
        
        up9 = Conv2DTranspose(2 * num_features, 
                              kernel_size = (2, 2), 
                              strides = (2, 2), 
                              **common_params)(conv8)
        up9 = concatenate([up9, conv1], axis=3)
        conv9_ = Conv2D(num_features, (3, 3), **common_params)(up9)
        conv9 = Conv2D(num_features, (3, 3), **common_params)(conv9_)
        
        conv10 = Conv2D(filters = 1, 
                        kernel_size = (1, 1), 
                        activation='sigmoid', 
                        kernel_initializer = kernel_init)(conv9)
        
        Cmodel = Model(inputs = [inputs], outputs = [conv10])

        return Cmodel

    @proc.func_timer  
    def create_unet_contours_model(
        self, 
        input_X: int = 128, 
        input_Y: int = 128,
        input_C: int = 1,
        num_features: int = 32,
        activation_fx: str = LeakyReLU(0.1),
        padding_method: str = 'same',
        kernel_init: str = 'random_uniform',
        with_bias: bool = False
    ) -> tf.keras.Model:
        common_params = {
            'activation': activation_fx,
            'padding': padding_method,
            'kernel_initializer': kernel_init,
            'use_bias': with_bias
        }
                
        inputs = Input((input_X, input_Y, input_C))
        
        conv1_ = Conv2D(num_features, 
                        kernel_size = (5, 5), 
                        input_shape = (input_X, input_Y, input_C), 
                        **common_params)(inputs)
        conv1 = Conv2D(num_features, (5, 5), **common_params)(conv1_)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2_ = Conv2D(2 * num_features, (5, 5), **common_params)(pool1)
        conv2 = Conv2D(2 * num_features, (5, 5), **common_params)(conv2_)
        pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)
        
        conv3_ = Conv2D(4 * num_features, (3, 3), **common_params)(pool2)
        conv3 = Conv2D(4 * num_features, (3, 3 ), **common_params)(conv3_)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4_ = Conv2D(8 * num_features, (3, 3), **common_params)(pool3)
        conv4 = Conv2D(8 * num_features, (3, 3), **common_params)(conv4_)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
        conv5_ = Conv2D(16 * num_features, (3, 3), **common_params)(pool4)
        conv5 = Conv2D(16 * num_features, (3, 3), **common_params)(conv5_)
        
        up6 = Conv2DTranspose(16 * num_features, 
                              kernel_size = (2, 2), 
                              strides = (2, 2), 
                              **common_params)(conv5)
        up6 = concatenate([up6, conv4], axis=3)
        conv6_ = Conv2D(8 * num_features, (3, 3), **common_params)(up6)
        conv6 = Conv2D(8 * num_features, (3, 3), **common_params)(conv6_)
        
        up7 = Conv2DTranspose(8 * num_features, 
                              kernel_size = (2, 2), 
                              strides = (2, 2), 
                              **common_params)(conv6)
        up7 = concatenate([up7, conv3], axis=3)
        conv7_ = Conv2D(4 * num_features, (3, 3), **common_params)(up7)
        conv7 = Conv2D(4 * num_features, (3, 3), **common_params)(conv7_)
        
        up8 = Conv2DTranspose(4 * num_features, 
                              kernel_size = (2, 2), 
                              strides = (2, 2), 
                              **common_params)(conv7)
        up8 = concatenate([up8, conv2], axis=3)
        conv8_ = Conv2D(2*num_features, (3, 3), **common_params)(up8)
        conv8 = Conv2D(2*num_features, (3, 3), **common_params)(conv8_)
        
        up9 = Conv2DTranspose(2 * num_features, 
                              kernel_size = (2, 2), 
                              strides = (2, 2), 
                              **common_params)(conv8)
        up9 = concatenate([up9, conv1], axis=3)
        conv9_ = Conv2D(num_features, (5, 5), **common_params)(up9)
        conv9 = Conv2D(num_features, (5, 5), **common_params)(conv9_)
        
        conv10 = Conv2D(filters = 1, 
                        kernel_size = (1, 1), 
                        activation='sigmoid', 
                        kernel_initializer = kernel_init)(conv9)
        
        model = Model(inputs = [inputs], outputs = [conv10])

        return model

    @proc.func_timer  
    def instance_segmentation(
        self,
        tile_array: np.ndarray,
        output_dir: Union[str, Path],
        file_name: str
    ) -> str:
        # make center detection predictions
        print("Predicting centers...")
        centers_data = self.centers_model.predict(tile_array)
        print("Center prediction shape:", centers_data.shape)

        # make segmentation prediction
        print("Predicting contours...")
        seg_data = self.contours_model.predict(tile_array)
        print("Contour prediction shape:", seg_data.shape)

        if centers_data is None or seg_data is None:
            print("Error: Prediction returned None.")
            return ""

        # generate seg and center maps as numpy arrays
        seg_map = seg_data.squeeze() * 255  # convert to 0-255 range
        centers_map = centers_data.squeeze() * 255  # convert to 0-255 range

        # creating output dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # saving contour and center maps as numpy arrays
        np.save(output_dir / f"{file_name}_segmentation_map.npy", seg_map)
        np.save(output_dir / f"{file_name}_centers_map.npy", centers_map)

        # cmg file
        num_nuclei = centers_data.shape[0]

        Z = np.array([0])
        h = fileio.create_header(num_nuclei, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z,
                                Z, Z, Z, Z, Z, Z, Z)
        cmg_file_path = output_dir / f"{file_name}.cmg"
        fileio.write_cmg(h, np.invert(tile_array), centers_data, cmg_file_path)

        return str(output_dir / f"{file_name}_segmentation_map.npy"), str(cmg_file_path)



    def __call__(
        self, 
        image_directory:Union[str,Path],
        image_filename:str
    ) -> None:
        return self.instance_segmentation(image_directory, image_filename)
    
# >>> EXPORT ALIASES <<< #
SUNet = SequentialUNetPipeline
Sunet = SequentialUNetPipeline






