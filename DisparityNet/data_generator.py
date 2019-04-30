from tensorflow import keras
import numpy as np
from IO import read
import json


class FlyingThings3D(keras.utils.Sequence):
    def __init__(self, data_list, batch_size=32, dim=(512, 512), input_channels=3, output_channels=1, shuffle=True):
        """Initializations"""
        self.dim = dim
        self.batch_size = batch_size
        self.data_list = data_list
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data_list))
        self.data_dir = 'data/flying_things_3d/frames_finalpass/'  # Insert path to frames_finalpass
        self.disparity_dir = 'data/flying_things_3d/disparity/'  # Insert path to disparity

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

            left[i, ] = read(self.data_dir+left_img)[:self.dim[0], :self.dim[1], :-1]
            right[i, ] = read(self.data_dir+right_img)[:self.dim[0], :self.dim[1], :-1]
            disparity[i, ] = read(self.disparity_dir+disparity_map)[:self.dim[0], :self.dim[1]].reshape((*self.dim, self.output_channels))

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
        return [left, right], disparity

    def __repr__(self):
        return '< FlyingThings3D with {} data points >'.format(len(self.data_list))


class SlidingTissues3D(keras.utils.Sequence):
    def __init__(self, data_list, batch_size=32, dim=(512, 512), input_channels=3, output_channels=1, shuffle=True):
        """Initializations"""
        self.dim = dim
        self.batch_size = batch_size
        self.data_list = data_list
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data_list))
        self.data_dir = 'data/sliding_tissues_3d/'  # Insert path to frames
        self.disparity_dir = 'data/sliding_tissues_3d/'  # Insert path to disparity

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
            disparity_map = str(left_img).replace('left.png', 'gt.pfm')

            left[i, ] = read(self.data_dir+left_img)[:self.dim[0], :self.dim[1], :]
            right[i, ] = read(self.data_dir+right_img)[:self.dim[0], :self.dim[1], :]
            disparity[i, ] = read(self.disparity_dir+disparity_map)[:self.dim[0], :self.dim[1]].reshape((*self.dim, self.output_channels))

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
        return [left, right], disparity

    def __repr__(self):
        return '< SlidingOrgans3D with {} data points >'.format(len(self.data_list))


# Loading list of data from JSON file
with open('data/flying_things_3d/output.json', 'r') as f:
    data_list_ft3d = json.load(f)

# Loading list of data from JSON file
with open('data/sliding_tissues_3d/output.json', 'r') as f:
    data_list_st3d = json.load(f)

# Parameters required by Generators
other_parameters = {
    'dim': (512, 512),
    'batch_size': 5,
    'input_channels': 3,
    'output_channels': 1,
}

# Creating Generator for Flying Things 3D
training_generator_ft3d = FlyingThings3D(data_list_ft3d['train'], shuffle=True, **other_parameters)
validation_generator_ft3d = FlyingThings3D(data_list_ft3d['validation'], shuffle=False, **other_parameters)

# Creating Generator for Sliding Organs 3D
training_generator_st3d = SlidingTissues3D(data_list_st3d['train'], shuffle=True, **other_parameters)
validation_generator_st3d = SlidingTissues3D(data_list_st3d['validation'], shuffle=False, **other_parameters)


# Testing the Generator
if __name__ == '__main__':
    print(training_generator_st3d)
    for epoch, data in enumerate(training_generator_st3d):
        print('Left:\t\t'+str(data[0][0].shape))
        print('Right:\t\t'+str(data[0][1].shape))
        print('Disparity:\t'+str(data[1].shape), end='\n\n')
        if epoch == 4:
            break
    print(training_generator_ft3d)
    for epoch, data in enumerate(training_generator_ft3d):
        print('Left:\t\t'+str(data[0][0].shape))
        print('Right:\t\t'+str(data[0][1].shape))
        print('Disparity:\t'+str(data[1].shape), end='\n\n')
        if epoch == 4:
            print('Terminated!')
            break
