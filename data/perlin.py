# https://github.com/VitjanZ/DRAEM/blob/main/perlin.py

import torch
import math
import numpy as np

def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    '''
    The code you provided generates a 2D Perlin noise texture using NumPy.
    Perlin noise is a gradient noise
    function commonly used in graphics and procedural generation for creating
    textures and natural phenomena
    INput
    shape: The desired output shape of the noise array.
    res: The resolution of the grid or the frequency of the noise.
    fade: A function to smooth the transition between grid points
     (default is a commonly used fade function).

    '''
    delta = (res[0] / shape[0], res[1] / shape[1])
    # delta represents the spacing between grid points in the output image.
    d = (shape[0] // res[0], shape[1] // res[1])
    # d is the number of times the gradient vectors will be repeated in
    # each dimension to cover the entire image.

    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # np.mgrid creates a grid of coordinates.
    # 0:res[0]:delta[0] creates a range from 0 to res[0] with steps of size delta[0].
    # This gives the x-coordinates.
    # 0:res[1]:delta[1] creates a range from 0 to res[1] with steps of size delta[1].
    # This gives the y-coordinates.
    # The grid is then adjusted to be in the range [0, 1] by applying modulo operation % 1.
    # returned a size of  (2, len(y-coordinates), len(x-coordinates)) so we transompose it
    # . The transpose(1, 2, 0) operation rearranges the dimensions so that the resulting
    # array has the shape (len(y-coordinates), len(x-coordinates), 2), where:
    #
    # The first dimension represents y-coordinates.
    # The second dimension represents x-coordinates.
    # The last dimension holds the coordinate pairs (x, y).
    # The reason for transposing is to rearrange the coordinate arrays into a format where each pixel
    # has a coordinate pair (x, y).
    # This format is more convenient for operations that require coordinate pairs.
    # Applying % 1 to the grid values normalizes them to the range [0, 1).
    # This operation ensures that the coordinates wrap around within the unit square.
    # This is important for noise generation as it ensures smooth transitions across the edges of the grid.
    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    # np.random.rand gives Random values in a given shape where the random samples are uniformly
    # distributed over 0 to 1
    # you multplie by 2 pi
    # Multiplying by 2 * math.pi converts these random numbers into angles in radians, ranging from 0 to 2π.
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    # np.cos(angles) calculates the cosine of each angle, producing the x-components of the gradient vectors.
    # np.sin(angles) calculates the sine of each angle, producing the y-components of the gradient vectors.
    # Each angle corresponds to a unit vector (cos(angle), sin(angle))
    # . These vectors are used to represent the gradient directions.
    # the gradients are created by converting these angles into unit vectors using cosine and sine.
    # For example, if np.cos(angles) and np.sin(angles) are both 2D arrays of shape (height, width),
    # then np.stack will combine them into a single array of shape (height, width, 2).
    # tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    # tile_grads returns a tiled version of the gradients for a specific grid section.
    # slice1 specifies the range of rows to select from the gradients array.
    # slice2 specifies the range of columns to select from the gradients array.
    # This extracts a subset of the gradients array:

    # slice1[0]:slice1[1] defines the row range.
    # slice2[0]:slice2[1] defines the column range.
    # For instance, if slice1 = [0, 2] and slice2 = [0, 2], this extracts the top-left 2x2 block of gradients.
    # np.repeat repeats the gradient vectors along the specified axis. axis=0 means repeating along the rows.
    # d[0] is the number of times to repeat the gradient vectors in the row direction.
    # For example, if the extracted subset is a 2x2 array
    # repeating it 3 times along the rows (d[0] = 3) would result in a 6x2 array.
    # THEN
    # The result from the previous np.repeat call is then repeated along the columns.
    # axis=1 means repeating along the columns.
    # d[1] is the number of times to repeat the gradient vectors in the column direction.
    # Continuing the example, if the intermediate result is a 6x2 array and you repeat it
    # 4 times along the columns (d[1] = 4), you get a 6x8 array.
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)
    # dot computes the dot product between the gradient vectors and the grid points shifted by
    # a specific amount.
    # defines a lambda function named dot that computes the dot product between a gradient vector
    # field and a shifted grid. This operation is a key part of the Perlin noise algorithm.
    #INPUT
    # # grad: This parameter is the gradient field array, which is a 2D array of gradient vectors
    # (each vector has components [gx, gy]).
    # shift: This is a vector that specifies how much to shift the grid coordinates.
    # It’s a tuple like [shift_x, shift_y].
    # PROCESSING
    # shape is the desired output shape of the noise
    # grid[:shape[0], :shape[1], 0] extracts the x-coordinates from the grid array for the region of interest.
    # grid[:shape[0], :shape[1], 1] extracts the y-coordinates from the grid array for the same region.
    # This slicing selects the portion of the grid that corresponds to the region covered by shape.
    # Add Shift:
    #
    # grid[:shape[0], :shape[1], 0] + shift[0] adds the x-component of the shift to the x-coordinates.
    # grid[:shape[0], :shape[1], 1] + shift[1] adds the y-component of the shift to the y-coordinates.
    # This combines the shifted x and y coordinates into a 3D array where the last dimension represents
    # the coordinate pairs [x', y']. SO you have (shape[0], shape[1], 2) containing the shifted coordinates.
    # The multiplication np.stack(..., axis=-1) * grad[:shape[0], :shape[1]]
    # computes the element-wise product of these arrays.
    # This means for each point (x', y') in the grid, the gradient vector (gx, gy)
    # is multiplied element-wise with the coordinates (x', y').
    # This sums the results along the last axis (which represents
    # the coordinate components and the gradient components).
    # As a result, for each grid point, you get a scalar value representing the dot product
    # of the gradient vector with the shifted grid coordinates.
    # The final result of the dot function is a 2D array with the shape (shape[0], shape[1]). Each element of this
    # array is a scalar value representing the dot product of the gradient vector with
    # the shifted coordinate vector at each grid point.
    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    # tile_grads([0, -1], [0, -1]):
    # Extracts the gradient field for the bottom-left corner of the grid cell.
    # The slice [0, -1] means it takes the gradients from the bottom row to the row before the last row,
    # and [0, -1] for the left column to the column before the last column.
    # Shift [0, 0]: No shift is applied.
    # Result: This computes the dot product at the bottom-left corner of the cell (coordinate [0, 0]).
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    # Apply the fade function to smooth the transitions between the grid points.

    # Interpolate to Get Final Noise Value
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)
    dot = lambda grad, shift: (
                torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])

    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise