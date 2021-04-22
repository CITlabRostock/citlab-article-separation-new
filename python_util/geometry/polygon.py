import math

from citlab_python_util.geometry import linear_regression as lin_reg
from citlab_python_util.geometry.point import rescale_points
from citlab_python_util.geometry.rectangle import Rectangle
from citlab_python_util.math.rounding import round_to_nearest_integer


class Polygon(object):

    def __init__(self, x_points=None, y_points=None, n_points=0):
        """ Constructs a new polygon. If x_points/y_points hold floats, these are converted to integers.

        :param x_points: list of x coordinates of the polygon
        :type x_points: list of int
        :param y_points: list of y coordinates of the polygon
        :type y_points: list of int
        :param n_points: total number of points in the polygon
        :type n_points: int
        :param bounds: bounding box of the polygon in shape of a rectangle
        :type bounds: Rectangle
        """
        if x_points is not None:
            if type(x_points) == list and all(type(x) == float for x in x_points):
                x_points = [int(x) for x in x_points]
            assert type(x_points) == list, "x_points has to be a list of ints"
            assert all(type(x) == int for x in x_points), "x_points has to be a list of ints"
            if n_points > len(x_points) or n_points > len(y_points):
                raise Exception("Bounds Error: n_points > len(x_points) or n_points > len(y_points)")

            self.x_points = x_points
        else:
            self.x_points = []

        if y_points is not None:
            if type(y_points) == list and all(type(y) == float for y in y_points):
                y_points = [int(y) for y in y_points]
            assert type(y_points) == list, "y_points has to be a list of ints"
            assert all(type(y) == int for y in y_points), "y_points has to be a list of ints"
            if n_points > len(x_points) or n_points > len(y_points):
                raise Exception("Bounds Error: n_points > len(x_points) or n_points > len(y_points)")

            self.y_points = y_points
        else:
            self.y_points = []

        assert type(n_points) == int, "n_points has to be int"
        if n_points < 0:
            raise Exception("Negative Size: n_points < 0")

        self.n_points = n_points
        self.bounds = None  # bounds of this polygon (Rectangle type !!!)

    def as_list(self):
        return list(zip(self.x_points, self.y_points))

    def rescale(self, scale):
        points_rescaled = rescale_points(self.as_list(), scale)
        x, y = zip(*points_rescaled)
        self.x_points = list(x)
        self.y_points = list(y)

        if self.bounds:
            self.calculate_bounds()

    def translate(self, delta_x, delta_y):
        """ Translates the vertices of this polygon by delta_x along the x axis and by delta_y along the y axis.

        :param delta_x: (int) amount to translate along the x axis
        :param delta_y: (int) amount to translate along the y axis
        """
        assert type(delta_x) == int, "delta_x has to be int"
        assert type(delta_y) == int, "delta_y has to be int"

        for i in range(self.n_points):
            self.x_points[i] += delta_x
            self.y_points[i] += delta_y

        if self.bounds is not None:
            self.bounds.translate(delta_x, delta_y)

    def calculate_bounds(self):
        """ Calculates the bounding box of points of the polygon. """

        bounds_min_x = min(self.x_points)
        bounds_min_y = min(self.y_points)

        bounds_max_x = max(self.x_points)
        bounds_max_y = max(self.y_points)

        self.bounds = Rectangle(bounds_min_x, bounds_min_y, width=bounds_max_x - bounds_min_x + 1,
                                height=bounds_max_y - bounds_min_y + 1)

    def update_bounds(self, x, y):
        """ Resizes the bounding box to accommodate the specified coordinates.

        :param x: (int) x coordinate
        :param y: (int) y coordinate
        """
        assert type(x) == int, "x has to be int"
        assert type(y) == int, "y has to be int"

        if x < self.bounds.x:
            self.bounds.width = self.bounds.width + (self.bounds.x - x)
            self.bounds.x = x
        else:
            self.bounds.width = max(self.bounds.width, x - self.bounds.x)

        if y < self.bounds.y:
            self.bounds.height = self.bounds.height + (self.bounds.y - y)
            self.bounds.y = y
        else:
            self.bounds.height = max(self.bounds.height, y - self.bounds.y)

    def add_point(self, x, y):
        """ Appends the specified coordinates to this polygon.

        :param x: (int) x coordinate of the added point
        :param y: (int) y coordinate of the added point
        """
        assert type(x) == int, "x has to be int"
        assert type(y) == int, "y has to be int"

        self.x_points.append(x)
        self.y_points.append(y)
        self.n_points += 1

        if self.bounds is not None:
            self.update_bounds(x, y)

    def get_bounding_box(self):
        """ Get the bounding box of this polygon (= smallest rectangle including this polygon).

        :return: (Rectangle) rectangle defining the bounds of this polygon
        """
        if self.n_points == 0:
            return Rectangle()

        if self.bounds is None:
            self.calculate_bounds()

        return self.bounds.get_bounds()

    def contains_point(self, point):
        """
        Check if point is contained in polygon.
        Run a semi-infinite ray horizontally (increasing x, fixed y) out from the test point,
        and count how many edges it crosses. At each crossing, the ray switches between inside and outside.
        This is called the Jordan curve theorem.
        :param point: tuple with x- and y-coordinates
        :return: bool, whether or not the point is contained in the polygon
        """
        # simple boundary check
        if not self.get_bounding_box().contains_point(point):
            return False

        is_inside = False
        point_x = point[0]
        point_y = point[1]
        for i in range(self.n_points):
            if (self.y_points[i] > point_y) is not (self.y_points[i - 1] > point_y):
                if point_x < (self.x_points[i - 1] - self.x_points[i]) * (point_y - self.y_points[i]) / \
                        (self.y_points[i - 1] - self.y_points[i]) + self.x_points[i]:
                    is_inside = not is_inside
        return is_inside


def blow_up(polygon):
    """ Takes a ``polygon`` as input and adds pixels to it according to the following rule. Consider the line between
    two adjacent pixels in the polygon (i.e., if connected via an egde). Then the method adds additional equidistand
    pixels lying on that line (if the value is double, convert to int), dependent on the x- and y-distance of the
    pixels.

    :param polygon: input polygon that should be blown up
    :type polygon: Polygon
    :return: blown up polygon
    """
    res = Polygon()

    for i in range(1, polygon.n_points, 1):
        x1 = polygon.x_points[i - 1]
        y1 = polygon.y_points[i - 1]
        x2 = polygon.x_points[i]
        y2 = polygon.y_points[i]
        diff_x = abs(x2 - x1)
        diff_y = abs(y2 - y1)
        # if (x1,y1) = (x2, y2)
        if max(diff_x, diff_y) < 1:
            if i == polygon.n_points - 1:
                res.add_point(x2, y2)
            continue

        res.add_point(x1, y1)
        if diff_x >= diff_y:
            for j in range(1, diff_x, 1):
                if x1 < x2:
                    xn = x1 + j
                else:
                    xn = x1 - j
                yn = int(round_to_nearest_integer(y1 + (xn - x1) * (y2 - y1) / (x2 - x1)))
                res.add_point(xn, yn)
        else:
            for j in range(1, diff_y, 1):
                if y1 < y2:
                    yn = y1 + j
                else:
                    yn = y1 - j
                xn = int(round_to_nearest_integer(x1 + (yn - y1) * (x2 - x1) / (y2 - y1)))
                res.add_point(xn, yn)
        if i == polygon.n_points - 1:
            res.add_point(x2, y2)

    return res


def thin_out(polygon, des_dist):
    """ Takes a (blown up) ``polygon`` as input and deletes pixels according to the destination distance (``des_dist``),
    s.t. two pixels have a max distance of ``des_dist``. An exception are polygons that are less than or equal to 20
    pixels.

    :param polygon: input (blown up) polygon that should be thinned out
    :param des_dist: max distance of two adjacent pixels
    :type polygon: Polygon
    :type des_dist: int
    :return: thinned out polygon
    """
    res = Polygon()

    if polygon.n_points <= 20:
        return polygon
    dist = polygon.n_points - 1
    min_pts = 20
    des_pts = max(min_pts, int(dist / des_dist) + 1)
    step = dist / (des_pts - 1)

    for i in range(des_pts - 1):
        idx = int(i * step)
        res.add_point(polygon.x_points[idx], polygon.y_points[idx])
    res.add_point(polygon.x_points[-1], polygon.y_points[-1])

    return res


def norm_poly_dists(poly_list, des_dist):
    """ For a given list of polygons ``poly_list`` calculate the corresponding normed polygons, s.t. every polygon has
    adjacent pixels with a distance of ~des_dist.

    :param poly_list: list of polygons
    :param des_dist: distance (measured in pixels) of two adjecent pixels in the destination polygon
    :type poly_list: list of Polygon
    :type des_dist: int
    :return: list of polygons
    """
    res = []

    for poly in poly_list:
        bb = poly.get_bounding_box()
        if bb.width > 100000 or bb.height > 100000:
            poly = Polygon([0], [0], 1)

        poly_blow_up = blow_up(poly)
        poly_thin_out = thin_out(poly_blow_up, des_dist)

        # to calculate the bounding box "get_bounds" must be executed
        poly_thin_out.get_bounding_box()
        res.append(poly_thin_out)

    return res


def calc_reg_line_stats(poly):
    """ Return the angle of baseline polygon ``poly`` and the intersection of the linear regression line with the
    y-axis.

    :param poly: input polygon
    :type poly: Polygon
    :return: angle of baseline and intersection of the linear regression line with the y-axis.
    """
    if poly.n_points <= 1:
        return 0.0, 0.0

    n = float("inf")
    if poly.n_points > 2:
        x_max = max(poly.x_points)
        x_min = min(poly.x_points)

        if x_max == x_min:
            m = float("inf")
        else:
            n, m = lin_reg.calc_line(poly.x_points, [-y for y in poly.y_points])
    else:
        x1, x2 = poly.x_points
        y1, y2 = [-y for y in poly.y_points]
        if x1 == x2:
            m = float("inf")
        else:
            m = (y2 - y1) / (x2 - x1)
            n = y2 - m * x2

    if m == float("inf"):
        angle = math.pi / 2
    else:
        angle = math.atan(m)

    # in special cases change the direction of the orientation (-> add pi to angle)
    if -math.pi / 2 < angle <= -math.pi / 4:
        if poly.y_points[0] > poly.y_points[-1]:
            angle += math.pi
    if -math.pi / 4 < angle <= math.pi / 4:
        if poly.x_points[0] > poly.x_points[-1]:
            angle += math.pi
    if math.pi / 4 < angle < math.pi / 2:
        if poly.y_points[0] < poly.y_points[-1]:
            angle += math.pi
    # Make sure that the angle is positive
    if angle < 0:
        angle += 2 * math.pi

    return angle, n


def string_to_poly(string_polygon):
    """ Parse the polygon represented by the string ``string_polygon`` and return a ``Polygon`` object.

    :param string_polygon: coordinates of a polygon given in string format: x1,y1;x2,y2;...;xn,yn
    :type string_polygon: str
    :return: Polygon object with the coordinates given in string_polygon
    """
    polygon = Polygon()
    points = string_polygon.split(";")

    if len(points) < 2:
        raise Exception("Wrong polygon string format.")

    for p in points:
        coord = p.split(",")
        if len(coord) < 2:
            raise Exception("Wrong polygon string format.")
        coord_x = int(coord[0])
        coord_y = int(coord[1])
        polygon.add_point(coord_x, coord_y)

    return polygon


def poly_to_string(polygon):
    """ Inverse method of ``string_to_poly``, taking a polygon as input and outputs a string holding the x,y coordinates
    of the points present in the polygon separated by semicolons ";".

    :param polygon: input polygon to be parsed
    :type polygon: Polygon
    :return: a string holding the x,y coordinates of the polygon in format: x1,y1;x2,y2;...;xn,yn
    """
    res = ""

    for x, y in zip(polygon.x_points, polygon.y_points):
        if len(res) != 0:
            res += ";"
        res += str(x) + "," + str(y)

    return res


def list_to_polygon_object(polygon_as_list):
    x, y = zip(*polygon_as_list)

    return Polygon(list(x), list(y), n_points=len(x))


def get_minimal_x(poly):
    return min(poly, key=lambda point: point[0])[0]


def get_minimal_y(poly):
    return min(poly, key=lambda point: point[1])[1]


def get_maximal_x(poly):
    return max(poly, key=lambda point: point[0])[0]


def get_maximial_y(poly):
    return max(poly, key=lambda point: point[1])[1]


def sort_ascending_by_x(polys):
    """ Sorts a list of polygons according to their x-values.

    :param polys: list of polygons given by a list of (x,y) tuples.
    :type polys: list of (list of (int, int))
    :return: sorted list of polygons given by a list of (x,y) tuples.
    """
    return sorted(polys, key=lambda poly: get_minimal_x(poly))


def sort_ascending_by_y(polys):
    """ Sorts a list of polygons according to their y-values.

    :param polys: list of polygons given by a list of (x,y) tuples.
    :type polys: list of (list of (int, int))
    :return: sorted list of polygons given by a list of (x,y) tuples.
    """
    return sorted(polys, key=lambda poly: get_maximial_y(poly))


def are_vertical_aligned(line1, line2, margin=20):
    line1_min_x = min(line1, key=lambda point: point[0])[0]
    line1_max_x = max(line1, key=lambda point: point[0])[0]
    line2_min_x = min(line2, key=lambda point: point[0])[0]
    line2_max_x = max(line2, key=lambda point: point[0])[0]

    if line2_min_x - margin <= line1_min_x <= line2_max_x and line2_min_x <= line1_max_x <= line2_max_x + margin:
        return True

    if line1_min_x - margin <= line2_min_x <= line1_max_x and line1_min_x <= line2_max_x <= line1_max_x + margin:
        return True

    if line1_min_x - margin < line2_min_x < line1_min_x + margin or line1_max_x - margin < line2_max_x < line1_max_x + margin:
        return True

    return False
