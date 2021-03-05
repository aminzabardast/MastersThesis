import argparse
import os
from .type_c import TypeC


def main():
    # Create a new network
    net = TypeC()

    # Train on the data
    net.predict(
        input_a_path=FLAGS.left,
        input_b_path=FLAGS.right,
        out_path=FLAGS.out,
        png_path=FLAGS.png
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l',
        '--left',
        type=str,
        required=True,
        help='Path to first image'
    )
    parser.add_argument(
        '-r',
        '--right',
        type=str,
        required=True,
        help='Path to second image'
    )
    parser.add_argument(
        '-o',
        '--out',
        type=str,
        required=True,
        help='Path to output result'
    )
    parser.add_argument(
        '-p',
        '--png',
        type=str,
        required=False,
        default='',
        help='Path to output png result'
    )
    FLAGS = parser.parse_args()

    # Verify arguments are valid
    if not os.path.exists(FLAGS.left):
        raise ValueError('image_a path must exist')
    if not os.path.exists(FLAGS.right):
        raise ValueError('image_b path must exist')
    if not os.path.isdir(FLAGS.out):
        raise ValueError('out directory must exist')
    main()
