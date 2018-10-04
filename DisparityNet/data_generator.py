from tensorflow import keras
import numpy as np
from IO import read
import json


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_list, batch_size=32, dim=(32, 32, 32), input_channels=3, output_channels=1, shuffle=True):
        """Initializations"""
        self.dim = dim
        self.batch_size = batch_size
        self.data_list = data_list
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.shuffle = shuffle
        self.indexes = None
        self.data_dir = 'data/flyingthings3d_data/frames_finalpass/'
        self.disparity_dir = 'data/flyingthings3d_data/disparity/'

    def on_epoch_end(self):
        """Update indexes after each epoch"""
        self.indexes = np.arange(len(self.data_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_imgs):
        """Generates data containing batch_size samples"""
        left = np.empty((self.batch_size, *self.dim, self.input_channels))
        right = np.empty((self.batch_size, *self.dim, self.input_channels))
        disparity = np.empty((self.batch_size, *self.dim, self.output_channels))

        for i, imgs in enumerate(list_imgs):
            left_img = imgs['left']
            right_img = imgs['right']
            disparity_map = str(left_img).replace('png', 'pfm')

            left[i, ] = read(self.data_dir+left_img).reshape((*self.dim, self.input_channels))
            right[i, ] = read(self.data_dir+right_img).reshape((*self.dim, self.input_channels))
            disparity[i, ] = read(self.disparity_dir+disparity_map).reshape((*self.dim, self.output_channels))

        return left, right, disparity

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.data_list) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Calculating the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Finding addresses of images
        list_imgs_temp = [self.data_list[k] for k in indexes]

        # Generating Data
        left, right, disparity = self.__data_generation(list_imgs_temp)
        return left, right, disparity


if __name__ == '__main__':
    json_list = json.load(open('output.json', 'r'))
    gen = DataGenerator(data_list=json_list['train'], dim=(540, 960), input_channels=4)
    gen.on_epoch_end()
    print(gen)
    for i in gen:
        print(i[0].shape)
        print(i[1].shape)
        print(i[2].shape)
