from src.utils import read_bag

# currently testing src.utils.read_bag
if __name__ == '__main__':
    import argparse
    import scipy.misc
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filepath",
        type=str,
        required=True,
        help="Path to the bag file")
    args = parser.parse_args()
    i = 0
    for color, depth in read_bag(args.filepath):
        i += 1
        if i % 100 == 50:
            print('Current frame:', i)
    print(i, 'frames read')
    scipy.misc.imshow(color)
    scipy.misc.imshow(depth)
