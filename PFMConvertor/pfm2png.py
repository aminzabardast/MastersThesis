import argparse
from IO import read
import matplotlib.pyplot as plt


# Options
version = 0.1


# Global Variables
parser = None
parsed_input = None
image = None


def parser_init():
    """Initiating parser and its settings"""
    global parser, parsed_input

    parser = argparse.ArgumentParser(description='''Converting PFM single chanel disparity map files 
    to color-coded PNG.''')
    parser.add_argument('--version',
                        action='version',
                        version='%(prog)s '+str(version))
    parser.add_argument('-i',
                        '--input-file',
                        action='store',
                        dest='pfm_file_path',
                        help='Path to PFM file',
                        required=True)
    parser.add_argument('-o',
                        '--output-file',
                        action='store',
                        dest='png_file_path',
                        default='output.png',
                        help='Path to PNG output.',
                        required=False)
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        default=False,
                        dest='is_verbose',
                        help='Print process report.')
    parser.add_argument('-c',
                        '--color-code',
                        action='store_true',
                        dest='is_color_coded',
                        default=False,
                        help='Color codes the output.')
    parsed_input = parser.parse_args()


def read_input_pfm():
    """Read PFM image into Numpy Array"""
    global image
    if file_format(parsed_input.pfm_file_path) != 'PFM':
        raise Exception('Input type is not PFM')
    image = read(parsed_input.pfm_file_path)


def write_output_png():
    """Writes Numpy Array to PNG image."""
    if file_format(parsed_input.png_file_path) != 'PNG':
        raise Exception('Output type is not PNG')
    plt.imsave(parsed_input.png_file_path, image, cmap=color_map())


def color_map():
    """Returning color map name"""
    return 'jet' if parsed_input.is_color_coded else 'gist_gray'


def file_format(image_path):
    """Returning File's Format"""
    return str(image_path).split('.')[-1].upper()


def reporter():
    """Simple Report"""
    if not parsed_input.is_verbose:
        return
    print('Converting PFM to PNG ...')


# Initiating Parser
parser_init()

# Reporting
reporter()

# Reading and Writing Images
read_input_pfm()
write_output_png()
