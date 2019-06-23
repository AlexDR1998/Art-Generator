# Art-Generator
Unconventional Image Processing

Doesn't do what most image processing programs do. If you want something for touching up photos look elsewhere. This mangles and abuses photos, often completely beyond recogniction. This is a program for turning ordinary photos into abstract art.



Use:

Run in terminal as: "python art_gen.py /path/to/image.jpg a b c"

where a,b,c are optional numbers to crop the image before processing. a and b select coordinates of center, c selects size of sqaure to crop. Included because a few of the methods (mainly the linear algebra ones) only work on square images

To select which method to apply to an image, edit the main method in art_gen.py



Code structure:

There are 2 files needed, image_manipulator.py and art_gen.py. The former includes a lot of lower level image manipulation functions, ranging from standard to just nonsensical. The latter includes more complex algorithms composed with these lower level functions.


Requirements:

Python 2.7, Numpy, Scipy, MatPlotLib

Also note this was written on Ubuntu, a few lines of file IO may need to be modified for use on windows/mac machines
