from data_generator import validation_generator_st3d, data_list_st3d
from .disparitynet_cs import DisparityNetCS
from IO import write
import matplotlib.pyplot as plt


def main():
    # Create a new network
    code = 'dn_cs_st3d_3'
    net = DisparityNetCS(code=code)
    predictions = net.predict_generator(validation_generator=validation_generator_st3d)

    path = 'data/sliding_tissues_3d/'

    for index, item in enumerate(data_list_st3d['validation']):
        try:
            print(path+item['left'][0:21]+code+'.pfm')
            write(path+item['left'][0:21]+code+'.pfm', predictions[index].reshape(512, 512))
            plt.imsave(path+item['left'][0:21]+code+'.png', predictions[index].reshape(512, 512), cmap='jet')
            print(predictions[index].shape)
        except:
            pass


if __name__ == '__main__':
    main()
