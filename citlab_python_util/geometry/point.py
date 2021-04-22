def rescale_points(points, scale):
    """ Take as input a list of points `points` in (x,y) coordinates and scale them according to the rescaling factor
     `scale`.

    :param points: list of points in (x,y) coordinates
    :type points: list of Tuple(int, int)
    :param scale: scaling factor
    :type scale: float
    :return: list of downscaled (x,y) points
    """
    return [(int(x * scale), int(y * scale)) for (x, y) in points]
