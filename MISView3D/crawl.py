import os
import time
import argparse
import json
import re

# Options
report_period = 1000

# Some global Variables
file = None
total_crawled = 0
start_time = time.time()
parsed_input = None
parser = None
data = {
    'train': [],
    'validation': []
}
valid_path = re.compile(pattern=r'\w+/\w\d/\d+', flags=re.IGNORECASE)


def crawler(image_directory):
    """Crawling images directory as a generator"""
    global total_crawled
    for path, _, files in os.walk(image_directory):
        total_crawled += 1
        path = path.replace(parsed_input.image_directory, '')
        is_valid = re.match(valid_path, path)
        if not is_valid:
            continue
        img_left = os.path.join(path, 'left.png')
        img_right = os.path.join(path, 'right.png')
        disparity = os.path.join(path, 'gt.pfm')
        yield (img_left, img_right, disparity)


def open_file():
    """Preparing output file"""
    global file, parsed_input
    file = open(parsed_input.output_file, 'w')


def close_file():
    """Closing output file"""
    global file
    file.close()


def update_data(crawler_item):
    """Updating the data dictionary"""
    img_left = crawler_item[0]
    img_right = crawler_item[1]
    if 'train' in img_left.lower():
        data['train'].append({
            'left': img_left,
            'right': img_right
        })
    else:
        data['validation'].append({
            'left': img_left,
            'right': img_right
        })


def writer():
    """Writing data into a JSON file"""
    global file, data
    try:
        json.dump(obj=data, fp=file, ensure_ascii=True)
    except Exception as e:
        print(e)


def crawling_reporter():
    """A reporter generating reports in each report period"""
    global parsed_input, report_period, total_crawled
    if not parsed_input.is_verbose:
        return
    blocks = int(total_crawled / report_period)
    print('Crawling... {}'.format('#' * blocks), end='\r')


def final_report():
    """A reporter generating final report"""
    global parsed_input, total_crawled
    if not parsed_input.is_verbose:
        return
    print('\nThe process took about %.2f seconds and %d data points are saved to %s\nTerminated!'
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
                        default='output.json',
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
    update_data(item)
    crawling_reporter()

# Write as JSON
writer()

# Closing file
close_file()

# Preparing Final Report
end_time = time.time()
final_report()
