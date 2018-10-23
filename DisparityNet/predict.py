from tensorflow.keras.models import load_model
from IO import read, write
import matplotlib.pyplot as plt
from metrics import bad_4_0, bad_2_0, bad_1_0, bad_4_0_np, bad_2_0_np, bad_1_0_np, bad_0_5_np
from tensorflow.keras.optimizers import Adam
import argparse
import os

# Removing Keras Log Outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Options
VERSION = 0.3
PARSER = None
PARSER_INPUT = None
INPUT_SHAPE = (1, 512, 512, 3)
DISPARITY_SHAPE = (512, 512)
MODEL = 'models/15.keras'


def parser_init():
    """Initiating parser and its settings"""
    global PARSER, PARSER_INPUT, is_verbose

    PARSER = argparse.ArgumentParser(description='''Converting PFM single chanel disparity map files 
    to color-coded PNG.''')
    PARSER.add_argument('--version',
                        action='version',
                        version='%(prog)s '+str(VERSION))
    PARSER.add_argument('-l',
                        '--left-image',
                        action='store',
                        dest='left_img_path',
                        help='Path to left image',
                        required=True)
    PARSER.add_argument('-r',
                        '--right-image',
                        action='store',
                        dest='right_img_path',
                        help='Path to right image',
                        required=True)
    PARSER.add_argument('-o',
                        '--output-image',
                        action='store',
                        dest='output_img',
                        default='output',
                        help='Output image name',
                        required=False)
    PARSER.add_argument('-f',
                        '--format',
                        action='store',
                        dest='output_format',
                        default='png',
                        help='Output image format',
                        required=False)
    PARSER.add_argument('-t',
                        '--truth',
                        action='store',
                        dest='truth_img',
                        help='Truth image path',
                        required=False)
    PARSER.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        default=False,
                        dest='is_verbose',
                        help='Print process report')
    PARSER_INPUT = PARSER.parse_args()
    is_verbose = PARSER_INPUT.is_verbose
    if is_verbose:
        print('Initializing ...', end='\r')


def read_images():
    # Loading data and reshaping to network input size
    global left_img, right_img, truth
    if is_verbose:
        print('Reading Images ...', end='\r')
    left_img = read(PARSER_INPUT.left_img_path)[:512, :512, 0:3].reshape(INPUT_SHAPE)
    right_img = read(PARSER_INPUT.right_img_path)[:512, :512, 0:3].reshape(INPUT_SHAPE)
    truth = False
    if PARSER_INPUT.truth_img:
        if is_verbose:
            print('Reading Truth ...', end='\r')
        truth = read(PARSER_INPUT.truth_img)[:512, :512].reshape(DISPARITY_SHAPE)


def load_network():
    # Loading the network and feeding the data
    if is_verbose:
        print('Loading Network ....', end='\r')
    global autoencoder
    autoencoder = load_model(MODEL, compile=False)  # Here should be an address for a pre-trained model

    # Optimizer
    optimizer = Adam()
    autoencoder.compile(optimizer=optimizer, loss='mse', metrics=[bad_1_0, bad_2_0, bad_4_0])


def predict():
    global disparity
    if is_verbose:
        print('Predicting ........', end='\r')
    disparity = autoencoder.predict(x=[left_img, right_img]).reshape(DISPARITY_SHAPE)
    formats = calculate_outputs_formats()

    for f in formats:
        if is_verbose:
            print('Saving Results ...', end='\r')
        if f == 'png':
            plt.imsave('{}.{}'.format(PARSER_INPUT.output_img, f), disparity, cmap='jet')
        elif f == 'pfm':
            write('{}.{}'.format(PARSER_INPUT.output_img, f), disparity)
    if truth.any():
        if is_verbose:
            print('Saving Truth ...', end='\r')
        plt.imsave('{}-truth.png'.format(PARSER_INPUT.output_img), truth, cmap='jet')


def calculate_outputs_formats():
    return str(PARSER_INPUT.output_format).lower().split('+')


def metrics():
    m4 = bad_4_0_np(truth, disparity)
    m2 = bad_2_0_np(truth, disparity)
    m1 = bad_1_0_np(truth, disparity)
    m0 = bad_0_5_np(truth, disparity)
    if is_verbose:
        print("Metrics are:\n\tBad 4.0: {}\n\tBad 2.0: {}\n\tBad 1.0: {}\n\tBad 0.5: {}\n".format(
            m4, m2, m1, m0))


def print_report():
    pass


parser_init()
read_images()
load_network()
predict()
metrics()
