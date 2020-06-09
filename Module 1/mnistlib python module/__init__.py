import mnist as mnistsrc
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import PIL
import PIL.ImageFilter
import numpy

'''def X():
    def work(img):
        return img
    return work'''

def Blur(radius):
    def work(img):
        return img.filter(PIL.ImageFilter.GaussianBlur(radius = radius))
    return work

def Rectangle(x_min, y_min, x_max, y_max):
    def work(img):
        nonlocal x_min, y_min, x_max, y_max
        size = img.size
        width = size[0]
        height = size[1]
        _x_min = round(x_min * width)
        _x_max = round(x_max * width)
        _y_min = round(y_min * height)
        _y_max = round(y_max * height)
        data = [_x_min, _y_min, _x_max, _y_max]
        img = img.transform(size, PIL.Image.EXTENT, data = data)
        return img
    return work

def __get_shift1_matrix(img_size):
    return numpy.array([ #shifts (0, 0) from left upper corner to the center of image
        [1, 0, img_size[0] * 0.5],
        [0, 1, img_size[1] * 0.5],
        [0, 0, 1]
    ])

def __get_shift2_matrix(img_size):
    return numpy.array([ #shifts (0, 0) from center to the left upper corner of image
        [1, 0, -img_size[0] * 0.5],
        [0, 1, -img_size[1] * 0.5],
        [0, 0, 1]
    ])

def Affine(matrix_part):
    def work(img):
        nonlocal matrix_part
        size = img.size
        matrix = numpy.array([
            matrix_part[0],
            matrix_part[1],
            [0, 0, 1]
        ])
        result = __get_shift1_matrix(size) \
            .dot(matrix) \
            .dot(__get_shift2_matrix(size))
        affine_data = result[0].tolist() + result[1].tolist()
        img = img.transform(size, PIL.Image.AFFINE, data = affine_data)
        return img
    return work

def Shear_x(coef):
    return Affine(
        numpy.array([
            [1, coef, 0],
            [0, 1, 0]
        ])
    )

def Shear_y(coef):
    return Affine(
        numpy.array([
            [1, 0, 0],
            [coef, 1, 0]
        ])
    )

def Scale_x(factor):
    return Affine(
        numpy.array([
            [1.0 / factor, 0, 0],
            [0, 1, 0]
        ])
    )

def Scale_y(factor):
    return Affine(
        numpy.array([
            [1, 0, 0],
            [0, 1.0 / factor, 0]
        ])
    )

def Scale(factor):
    return Affine(
        numpy.array([
            [1.0 / factor, 0, 0],
            [0, 1.0 / factor, 0]
        ])
    )

def Rotate(angle):
    return Affine(
        numpy.array([
            [math.cos(angle), math.sin(angle), 0],
            [-math.sin(angle), math.cos(angle), 0]
        ])
    )

def Perspective(upper_left, lower_left, lower_right, upper_right): #in percents
    def work(img):
        nonlocal upper_left, lower_left, lower_right, upper_right
        size = img.size
        width = size[0]
        height = size[1]
        def calc(point_percents):
            return (round(point_percents[0] * width), round(point_percents[1] * height))
        _upper_left = calc(upper_left)
        _lower_left = calc(lower_left)
        _lower_right = calc(lower_right)
        _upper_right = calc(upper_right)
        img = img.transform(size, PIL.Image.QUAD,
            data = (
                _upper_left[0], _upper_left[1],
                _lower_left[0], _lower_left[1],
                _lower_right[0], _lower_right[1],
                _upper_right[0], _upper_right[1]
            )
        )
        return img
    return work

def Sharpen():
    def work(img):
        for i in range(5):
            img = img.filter(PIL.ImageFilter.GaussianBlur(radius = 3))
            img = img.filter(PIL.ImageFilter.UnsharpMask(radius=1, percent=1000, threshold=0))
        img = img.filter(PIL.ImageFilter.MinFilter(size=5))
        img = img.filter(PIL.ImageFilter.MaxFilter(size=5))
        img = img.filter(PIL.ImageFilter.UnsharpMask(radius=5, percent=1000, threshold=3))
        img = img.filter(PIL.ImageFilter.GaussianBlur(radius = 3))
        img = img.filter(PIL.ImageFilter.MinFilter(size=5))
        img = img.filter(PIL.ImageFilter.UnsharpMask(radius=5, percent=1000, threshold=3))
        return img
    return work

def Deform(grid_size, distortion):
    def work(img):
        #nonlocal seed
        #if seed != None:
        #    random.seed(seed)
        nonlocal grid_size, distortion
        def random_ball(num_points, dimension, radius = 1):
            from numpy import random, linalg
            # First generate random directions by normalizing the length of a
            # vector of random-normal values (these distribute evenly on ball).
            random_directions = random.normal(size=(dimension,num_points))
            random_directions /= linalg.norm(random_directions, axis=0)
            # Second generate a random radius with probability proportional to
            # the surface area of a ball with a given radius.
            random_radii = random.random(num_points) ** (1/dimension)
            # Return the list of random (direction & length) points.
            return radius * (random_directions * random_radii).T

        def get_random_mesh(img_size, steps, distortion):
            if (distortion < 0 or distortion > 0.5):
                raise "distortion should be between 0 and 0.5"
            if (steps < 1 or steps > 10):
                raise "steps should be between 1 and 10"
            points = numpy.zeros((steps + 1, steps + 1, 2), dtype = int)
            points_deformed = numpy.copy(points)
            dx = round(img_size[0] / steps)
            dy = round(img_size[1] / steps)
            for step_x in range(steps + 1):
                for step_y in range(steps + 1):
                    x = step_x * dx
                    y = step_y * dy
                    points[step_x][step_y][0] = x
                    points[step_x][step_y][1] = y
                    rand_xy = random_ball(1, 2, dx * distortion)[0]
                    rand_dx = rand_xy[0]
                    rand_dy = rand_xy[1] * dy / dx
                    points_deformed[step_x][step_y][0] = x + round(rand_dx)
                    points_deformed[step_x][step_y][1] = y + round(rand_dy)
            mesh = []
            for step_x in range(steps):
                for step_y in range(steps):
                    bbox = (
                        points[step_x][step_y][0],
                        points[step_x][step_y][1],
                        points[step_x + 1][step_y + 1][0],
                        points[step_x + 1][step_y + 1][1]
                    )
                    upper_left = points_deformed[step_x][step_y]
                    lower_left = points_deformed[step_x][step_y + 1]
                    lower_right = points_deformed[step_x + 1][step_y + 1]
                    upper_right = points_deformed[step_x + 1][step_y]
                    quad = (
                        upper_left[0], upper_left[1], #upper left
                        lower_left[0], lower_left[1], #lower left
                        lower_right[0], lower_right[1], #lower right
                        upper_right[0], upper_right[1] #upper right
                    )
                    mesh.append((bbox, quad))
            return mesh
        mesh = get_random_mesh(img.size, grid_size, distortion)
        img = img.transform(img.size, PIL.Image.MESH, data = mesh)
        return img
    return work

def np_to_pil(img):
    return PIL.Image.fromarray(img , 'L')

def pil_to_np(img):
    return numpy.reshape(numpy.asarray(img.getdata(), dtype = numpy.uint8, order = 'F'), (img.size[1], img.size[0]))

def edit(img, *filters, scaling = 4, demo = False):
    '''def draw_grid(img, step, clear = False):
        for x in range(img.size[0]):
            for y in range(img.size[1]):
                if (x % step <= step - 2 or y % step <= step - 2):
                    img.putpixel((x, y), 255)
                elif clear:
                    img.putpixel((x, y), 0)'''
    if (type(scaling) != int):
        raise TypeError("scaling should be integer more or equal to 1")
    if (scaling < 1):
        raise ValueError("scaling should be integer more or equal to 1")
    if (len(filters) == 0):
        raise ValueError("empty list of filters")
    is_np = img.__class__ == numpy.ndarray
    is_pil = img.__class__ == PIL.Image.Image
    if is_np:
        img = np_to_pil(img)
    elif not is_pil:
        raise TypeError("img type is unknown, use 2-dim np array or pil grayscale image")
    initial_size = img.size
    scaled_size = (initial_size[0] * scaling, initial_size[1] * scaling)
    img = img.resize(scaled_size, PIL.Image.LANCZOS)
    ####################
    '''if demo:
        draw_grid(img, 5 * scaling, False)'''
    for f in filters:
        img = f(img)
    ####################
    if not demo:
        img = img.resize(initial_size, PIL.Image.LANCZOS)
    if is_np:
        img = pil_to_np(img)
    return img

def fits(img, threshold = 64):
    if img.__class__ == numpy.ndarray:
        #should change X and Y?
        for x in range(img.shape[0]):
            if img[x][0] > threshold or img[x][img.shape[1] - 1] > threshold:
                return False
        for y in range(img.shape[1]):
            if img[0][y] > threshold or img[img.shape[0] - 1][y] > threshold:
                return False
        return True
    elif img.__class__ == PIL.Image.Image:
        for x in range(img.size[0]):
            if img.getpixel((x, 0)) > threshold or img.getpixel((x, img.size[1] - 1)) > threshold:
                return False
        for y in range(img.size[1]):
            if img.getpixel((0, y)) > threshold or img.getpixel((img.size[0] - 1, y)) > threshold:
                return False
        return True
    else:
        raise TypeError("img type is unknown, use 2-dim np array or pil grayscale image")

def draw(*images, smooth = False, columns = 4):
    rows = math.ceil(len(images) / columns)
    fig = plt.figure()
    fig.set_figwidth(16)
    fig.set_figheight(16 * rows / columns)
    for i in range(len(images)):
        fig.add_subplot(rows, columns, i + 1)
        interp = 'bicubic' if smooth else 'nearest'
        img = images[i]
        if img.__class__ == PIL.Image.Image:
            img = pil_to_np(img)
        plt.imshow(img, cmap = plt.cm.binary, interpolation = interp)
    plt.show()

x_train = None
x_train_flat = None
y_train = None

x_val = None
x_val_flat = None
y_val = None

def init():
    global x_train, x_val, y_train, y_val, x_train_flat, x_val_flat
    #(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    x_train = mnistsrc.train_images()
    x_val = mnistsrc.test_images()
    y_train = mnistsrc.train_labels()
    y_val = mnistsrc.test_labels()
    #dir = os.getcwd()
    #x_train = mnistsrc.download_and_parse_mnist_file('train-images-idx3-ubyte.gz', target_dir = dir)
    #x_val = mnistsrc.download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz', target_dir = dir)
    #y_train = mnistsrc.download_and_parse_mnist_file('train-labels-idx1-ubyte.gz', target_dir = dir)
    #y_val = mnistsrc.download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz', target_dir = dir)
    x_train_flat = numpy.array([x.reshape(-1).astype(float) for x in x_train])
    x_val_flat = numpy.array([x.reshape(-1).astype(float) for x in x_val])