# Chlorophyll level estimation

## Setup

To install the required python libraries for this project run `setup.sh`.

```
./setup.sh
```

## Functionalities

### Image color

The file `image_color.py` is used to get the representative color of an image. The first argument is the path to the image in use. The second argument specifies the output format of the color chosen from the image. This is an optional argument, its default value is LAB color format.

Note that `image_color.py` considers all the pixels from an image. Running the program with big pictures might take a long time.

#### Examples

Output the color of an image in RGB format.

```
python3 image_color.py data/input/samples/cutted_image.png -f RGB
```

### Image box color

The procedure behind `image_box_color.py` is similar to that of `image_color.py`, but receives additional parameters to define a box from which the color is chosen. The first argument is the path to the image, followed by the X and Y coordinates of a box within the image. The next arguments are the box width and height. Optionally, you can provide the color format type.

#### Examples

Define a box in the middle of an image and print its color in LAB format.

```
python3 image_box_color.py data/input/samples/full_image.jpeg 1900 1250 200 200 -f LAB
```
