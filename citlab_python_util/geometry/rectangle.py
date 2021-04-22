import sys


class Rectangle(object):

    def __init__(self, x=0, y=0, width=0, height=0):
        """ Constructs a new rectangle.

        :param x: (int) x coordinate of the upper left corner of the rectangle
        :param y: (int) y coordinate of the upper left corner of the rectangle
        :param width: (int) width of the rectangle
        :param height: (int) height of the rectangle
        """
        assert type(x) == int, "x has to be int"
        assert type(y) == int, "y has to be int"
        assert type(width) == int, "width has to be int"
        assert type(height) == int, "height has to be int"

        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_bounds(self):
        """ Get the bounding rectangle of this rectangle.

        :return: (Rectangle) bounding rectangle
        """
        return Rectangle(self.x, self.y, width=self.width, height=self.height)

    def set_bounds(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_vertices(self):
        """ Get the four vertices of this rectangle.

        :return: List of four tuples with x- and y-coordinates
        """
        v1 = (self.x, self.y)
        v2 = (self.x + self.width, self.y)
        v3 = (self.x + self.width, self.y + self.height)
        v4 = (self.x, self.y + self.height)
        return [v1, v2, v3, v4]

    def contains_point(self, point):
        """ Check if a point is contained in this rectangle.

        :param point: tuple with x- and y-coordinates
        :return: bool, whether or not the point is contained in the rectangle
        """
        px = point[0]
        py = point[1]
        return self.x < px < self.x + self.width and self.y < py < self.y + self.height

    def contains_point_on_boundary(self, point):
        """ Check if a point is lying on the rectangle border.

        :param point: tuple with x- and y-coordinates
        :return: bool, whether or not the point is lying on the rectangle border
        """
        px = point[0]
        py = point[1]

        is_on_vertical_edge = (px == self.x or px == self.x + self.width) and self.y <= py <= self.y + self.height
        is_on_horizontal_edge = (py == self.y or py == self.y + self.height) and self.x <= px <= self.x + self.width

        return is_on_horizontal_edge and is_on_vertical_edge

    def translate(self, dx, dy):
        """ Translates this rectangle the indicated distance, to the right along the x coordinate axis, and downward
        along the y coordinate axis.

        :param dx: (int) amount to translate along the x axis
        :param dy: (int) amount to translate along the y axis
        """
        assert type(dx) == int, "dx has to be int"
        assert type(dy) == int, "dy has to be int"

        old_v = self.x
        new_v = old_v + dx

        if dx < 0:
            # moving leftward
            if new_v > old_v:
                # negative overflow
                if self.width >= 0:
                    self.width += new_v - (-sys.maxsize - 1)

                new_v = -sys.maxsize - 1
        else:
            # moving rightward or staying still
            if new_v < old_v:
                # positive overflow
                if self.width >= 0:
                    self.width += new_v - sys.maxsize

                    if self.width < 0:
                        self.width = sys.maxsize

                new_v = sys.maxsize

        self.x = new_v

        old_v = self.y
        new_v = old_v + dy

        if dy < 0:
            # moving upward
            if new_v > old_v:
                # negative overflow
                if self.height >= 0:
                    self.height += new_v - (-sys.maxsize - 1)

                new_v = -sys.maxsize - 1
        else:
            # moving downward or staying still
            if new_v < old_v:
                # positive overflow
                if self.height >= 0:
                    self.height += new_v - sys.maxsize

                    if self.height < 0:
                        self.height = sys.maxsize

                new_v = sys.maxsize

        self.y = new_v

    def intersection(self, r):
        """ Computes the intersection of this rectangle with the specified rectangle.

        :param r: (Rectangle) specified rectangle
        :return: (Rectangle) a new rectangle presenting the intersection of the two rectangles
        """
        assert type(r) == Rectangle, "r has to be Rectangle"

        tx1 = self.x
        ty1 = self.y
        tx2 = tx1 + self.width
        ty2 = ty1 + self.height

        rx1 = r.x
        ry1 = r.y
        rx2 = rx1 + r.width
        ry2 = ry1 + r.height

        if tx1 < rx1:
            tx1 = rx1
        if ty1 < ry1:
            ty1 = ry1
        if tx2 > rx2:
            tx2 = rx2
        if ty2 > ry2:
            ty2 = ry2

        # width of the intersection rectangle
        tx2 -= tx1
        # height of the intersection rectangle
        ty2 -= ty1
        # tx2, ty2 might underflow
        if tx2 < -sys.maxsize - 1:
            tx2 = -sys.maxsize - 1
        if ty2 < -sys.maxsize - 1:
            ty2 = -sys.maxsize - 1

        return Rectangle(tx1, ty1, width=tx2, height=ty2)

    def contains_rectangle(self, r):
        """ Checks if the Rectangle object contains another Rectangle object ``r``.

        :param r: rectangle to check if it lies in the current Rectangle object
        :type r: Rectangle
        :return: True if ``r`` is contained, False otherwise.
        """
        vertices_r = r.get_vertices()
        for v in vertices_r:
            if not (self.contains_point(v) or self.contains_point_on_boundary(v)):
                return False

        return True

    def lies_above_of(self, r):
        """ Checks if the Rectangle lies above of Rectangle ``r``."""
        if self.y + self.height < r.y:
            return True
        return False

    def lies_below_of(self, r):
        """ Checks if the Rectangle lies below of Rectangle ``r``."""
        if self.y < r.y + r.height:
            return True
        return False

    def lies_left_of(self, r):
        """ Checks if the Rectangle lies left of Rectangle ``r``."""
        if self.x > r.x + r.width:
            return True
        return False

    def lies_right_of(self, r):
        """ Checks if the Rectangle lies right of Rectangle ``r``."""
        if self.x + self.width < r.x:
            return True
        return False

    def get_gap_to(self, r):
        intersection = self.intersection(r)
        if intersection.width > 0 and intersection.height > 0:
            return Rectangle(0, 0, 0, 0)
        if intersection.width > 0:
            return Rectangle(intersection.x, intersection.y - abs(intersection.height), intersection.width,
                             abs(intersection.height))
        if intersection.height > 0:
            return Rectangle(intersection.x - abs(intersection.width), intersection.y, abs(intersection.width),
                             intersection.height)
        else:
            return Rectangle(intersection.x - abs(intersection.width), intersection.y - abs(intersection.height),
                             abs(intersection.width), abs(intersection.height))

    def rescale(self, scaling_factor):
        if scaling_factor * self.width < 1 or scaling_factor * self.height < 1:
            return None
        self.x = int(scaling_factor * self.x)
        self.y = int(scaling_factor * self.y)
        self.width = int(scaling_factor * self.width)
        self.height = int(scaling_factor * self.height)
