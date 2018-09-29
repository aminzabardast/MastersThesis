import os
import time
import argparse


# Options
report_period = 1000

# Some global Variables
file = None
total_crawled = 0
start_time = time.time()
parsed_input = None
parser = None


def crawler(image_directory):
    """Crawling images directory as a generator"""
    global total_crawled
    for path, _, files in os.walk(image_directory):
        for name in files:
            if 'right' in path:
                continue
            total_crawled += 1
            path = path.replace(parsed_input.image_directory, '')
            img_left = os.path.join(path, name)
            img_right = img_left.replace('left', 'right')
            disparity_left = img_left.replace('png', 'pfm')
            disparity_right = img_right.replace('png', 'pfm')
            yield (img_left, img_right, disparity_left, disparity_right)


def open_file():
    """Preparing output file"""
    global file, parsed_input
    file = open(parsed_input.output_file, 'w')
    writer(['left_image', 'right_image', 'left_disparity', 'right_disparity'])


def close_file():
    """Closing output file"""
    global file
    file.close()


def writer(crawler_item):
    """Writing parsed_input into file"""
    global file
    try:
        file.write(','.join(crawler_item)+'\n')
    except Exception as e:
        print(e)


def crawling_reporter():
    """A reporter generating reports in each report period"""
    global parsed_input, report_period, total_crawled
    if not parsed_input.is_verbose:
        return
    blocks = int(total_crawled / report_period)
    print('Crawling... {}'.format('#'*blocks), end='\r')


def final_report():
    """A reporter generating final report"""
    global parsed_input, total_crawled
    if not parsed_input.is_verbose:
        return
    print('\nThe process took about %.2f seconds and %d lines are saved to %s.\nTerminated!'
          % (end_time - start_time, total_crawled, parsed_input.output_file))
    

def parser_init():
    """Initiating parser and its settings"""
    global parser, parsed_input
    parser = argparse.ArgumentParser(description='Creating a list of data compatible with Keras Data Generator.')

    parser.add_argument('--version',
                        action='version',
                        version='%(prog)s 0.1')

    parser.add_argument('-i',
                        '--image-dir',
                        action='store',
                        dest='image_directory',
                        help='Directory containing stereo images.',
                        required=True)

    parser.add_argument('-o',
                        '--output',
                        action='store',
                        dest='output_file',
                        default='output.csv',
                        help='Designating output file name.',
                        required=False)

    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        default=False,
                        dest='is_verbose',
                        help='Print process statistics to the screen.')

    # Parsing input arguments
    parsed_input = parser.parse_args()

    # Appending / to directory path
    if parsed_input.image_directory[-1] is not '/':
        parsed_input.image_directory += '/'


# Initiating Parser
parser_init()

# Opening output file
open_file()

# Writing data
for item in crawler(parsed_input.image_directory):
    writer(item)
    crawling_reporter()

# Closing file
close_file()

# Preparing Final Report
end_time = time.time()
final_report()
