import argparse
from PIL import Image

def split_image(input_image, chunk_size):
    """
    Split an image into multiple smaller chunks of a specified size
    """
    img = Image.open(input_image)
    width, height = img.size
    chunk_width, chunk_height = chunk_size

    # calculate the number of chunks to be generated
    chunks = [(width // chunk_width + 1) * (height // chunk_height + 1)]

    # iterate over the rows and columns
    for row in range(0, height, chunk_height):
        for col in range(0, width, chunk_width):
            # crop the image
            box = (col, row, col + chunk_width, row + chunk_height)
            yield img.crop(box)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split an image into smaller chunks')
    parser.add_argument('input_image', type=str, help='Path to the input image')
    parser.add_argument('chunk_width', type=int, help='Width of each chunk in pixels')
    parser.add_argument('chunk_height', type=int, help='Height of each chunk in pixels')
    args = parser.parse_args()

    chunk_size = (args.chunk_width, args.chunk_height)

    # generate the chunks
    for i, chunk in enumerate(split_image(args.input_image, chunk_size)):
        chunk.save(f"chunk_{i}.jpg")