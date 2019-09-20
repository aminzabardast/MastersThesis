from data_generator import validation_generator_misv3d, data_list_misv3d
from .type_s import TypeS
from IO import write
import matplotlib.pyplot as plt


def main():
    # Create a new network
    code = 'dn_s_st3d_1'
    net = TypeS(code=code)
    predictions = net.predict_generator(validation_generator=validation_generator_misv3d)

    path = 'data/sliding_tissues_3d/'

    for index, item in enumerate(data_list_misv3d['validation']):
        try:
            print(path+item['left'][0:21]+code+'.pfm')
            write(path+item['left'][0:21]+code+'.pfm', predictions[index].reshape(512, 512))
            plt.imsave(path+item['left'][0:21]+code+'.png', predictions[index].reshape(512, 512), cmap='jet')
            print(predictions[index].shape)
        except:
            pass


if __name__ == '__main__':
    main()
