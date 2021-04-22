import functools

import math
import numpy as np
from collections import Counter
from scipy.spatial import Delaunay

from citlab_python_util.geometry.polygon import calc_reg_line_stats, Polygon, norm_poly_dists
from citlab_python_util.geometry.rectangle import Rectangle


def merge_rectangles(rectangle_list):
    """

    :param rectangle_list:
    :type rectangle_list: list of Rectangle
    :return: minimal Rectangle object that holds all rectangles in rectangle_list
    """

    min_x = min(rectangle_list, key=lambda rectangle: rectangle.x).x
    max_x = max(rectangle_list, key=lambda rectangle: rectangle.x + rectangle.width).get_vertices()[1][0]
    min_y = min(rectangle_list, key=lambda rectangle: rectangle.y).y
    max_y = max(rectangle_list, key=lambda rectangle: rectangle.y + rectangle.height).get_vertices()[2][1]

    return Rectangle(min_x, min_y, max_x - min_x, max_y - min_y)


def check_intersection(line_1, line_2):
    """ Checks if two line segments `line1` and `line2` intersect. If they do so, the function returns the intersection
    point as [x,y] coordinate (special case for overlapping ["inf", "inf"]), otherwise `None`.

    :param line_1: list containing the x- and y-coordinates as [[x1,x2],[y1,y2]]
    :param line_2: list containing the x- and y-coordinates as [[x1,x2],[y1,y2]]
    :return: intersection point [x,y] if the line segments intersect, None otherwise
    """
    x_points1, y_points1 = line_1
    x_points2, y_points2 = line_2

    # consider vector form (us + s*vs = u + t*v)
    us = [x_points1[0], y_points1[0]]
    vs = [x_points1[1] - x_points1[0], y_points1[1] - y_points1[0]]

    u = [x_points2[0], y_points2[0]]
    v = [x_points2[1] - x_points2[0], y_points2[1] - y_points2[0]]

    A = np.array([vs, [-v[0], -v[1]]]).transpose()
    b = np.array([u[0] - us[0], u[1] - us[1]])

    rank_A = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(np.c_[A, b])

    # no solution => parallel
    if rank_A != rank_Ab:
        return None

    # infinite solutions => one line is the multiple of the other
    if rank_A == rank_Ab == 1:
        # check if there is an overlap
        # us + s*vs = u
        s1 = (u[0] - us[0]) / vs[0]
        s2 = (u[1] - us[1]) / vs[1]
        if s1 == s2:
            if 0 < s1 < 1:
                return ["inf", "inf"]
            elif s1 == 0 or s1 == 1:
                return [us[0] + s1 * vs[0], us[1] + s1 * vs[1]]

        # us + s*vs = v
        s1 = (v[0] - us[0]) / vs[0]
        s2 = (v[1] - us[1]) / vs[1]
        if s1 == s2:
            if 0 < s1 < 1:
                return ["inf", "inf"]
            elif s1 == 0 or s1 == 1:
                return [us[0] + s1 * vs[0], us[1] + s1 * vs[1]]

        # otherwise there is no overlap and no intersection
        return None

    [s, t] = np.linalg.inv(A).dot(b)

    if not (0 <= s <= 1 and 0 <= t <= 1):
        return None

    return [us[0] + s * vs[0], us[1] + s * vs[1]]


def ortho_connect(rectangles):
    """
    2D Orthogonal Connect-The-Dots, see
    http://cs.smith.edu/~jorourke/Papers/OrthoConnect.pdf

    :param rectangles: list of Rectangle objects
    :return: list of surrounding polygons over rectangles
    """
    assert type(rectangles) == list
    assert all([isinstance(rect, Rectangle) for rect in rectangles])

    # Go over vertices of each rectangle and only keep shared vertices
    # if they are shared by an odd number of rectangles
    points = set()
    for rect in rectangles:
        for pt in rect.get_vertices():
            if pt in points:
                points.remove(pt)
            else:
                points.add(pt)
    points = list(points)

    def y_then_x(a, b):
        if a[1] < b[1] or (a[1] == b[1] and a[0] < b[0]):
            return -1
        elif a == b:
            return 0
        else:
            return 1

    sort_x = sorted(points)
    sort_y = sorted(points, key=functools.cmp_to_key(y_then_x))

    edges_h = {}
    edges_v = {}

    # go over rows (same y-coordinate) and draw edges between vertices 2i and 2i+1
    i = 0
    while i < len(points):
        curr_y = sort_y[i][1]
        while i < len(points) and sort_y[i][1] == curr_y:
            edges_h[sort_y[i]] = sort_y[i + 1]
            edges_h[sort_y[i + 1]] = sort_y[i]
            i += 2

    # go over columns (same x-coordinate) and draw edges between vertices 2i and 2i+1
    i = 0
    while i < len(points):
        curr_x = sort_x[i][0]
        while i < len(points) and sort_x[i][0] == curr_x:
            edges_v[sort_x[i]] = sort_x[i + 1]
            edges_v[sort_x[i + 1]] = sort_x[i]
            i += 2

    # Get all the polygons
    all_polygons = []
    while edges_h:
        # We can start with any point
        polygon = [(edges_h.popitem()[0], 0)]
        while True:
            curr, e = polygon[-1]
            if e == 0:
                next_vertex = edges_v.pop(curr)
                polygon.append((next_vertex, 1))
            else:
                next_vertex = edges_h.pop(curr)
                polygon.append((next_vertex, 0))
            if polygon[-1] == polygon[0]:
                # Closed polygon
                polygon.pop()
                break
        # Remove implementation-markers from the polygon
        poly = [point for point, _ in polygon]
        for vertex in poly:
            if vertex in edges_h:
                edges_h.pop(vertex)
            if vertex in edges_v:
                edges_v.pop(vertex)

        poly_xs, poly_ys = zip(*poly)
        all_polygons.append(Polygon(list(poly_xs), list(poly_ys), len(poly_xs)))

    # Remove polygons contained in other polygons
    final_polygons = all_polygons.copy()
    if len(all_polygons) > 1:
        for poly in all_polygons:
            tmp_polys = all_polygons.copy()
            tmp_polys.remove(poly)
            # Only need to check if one point of the polygon is contained in another polygon
            # (By construction, the entire polygon is contained then)
            for tpoly in tmp_polys:
                if tpoly.contains_point((poly.x_points[0], poly.y_points[0])):
                    final_polygons.remove(poly)

    return final_polygons


def get_orientation_rectangles(point, dims=(600, 300, 600, 300), offset=0):
    # Verticals are North and South
    height_v = dims[0]
    width_v = dims[1]
    # Horizontals are East and West
    height_h = dims[2]
    width_h = dims[3]
    pt_x = point[0]
    pt_y = point[1]
    rect_n = Rectangle(pt_x - width_v // 2, pt_y - height_v, width_v, height_v)
    rect_n.translate(dx=0, dy=offset)
    rect_s = Rectangle(pt_x - width_v // 2, pt_y, width_v, height_v)
    rect_s.translate(dx=0, dy=-offset)
    rect_e = Rectangle(pt_x, pt_y - height_h // 2, width_h, height_h)
    rect_e.translate(dx=-offset, dy=0)
    rect_w = Rectangle(pt_x - width_h, pt_y - height_h // 2, width_h, height_h)
    rect_w.translate(dx=offset, dy=0)
    return {'n': rect_n, 'e': rect_e, 's': rect_s, 'w': rect_w}


def get_orientation_cones(point, dims=(600, 300, 600, 300), offset=0):
    # Verticals are North and South
    height_v = dims[0]
    width_v = dims[1]
    # Horizontals are East and West
    height_h = dims[2]
    width_h = dims[3]
    pt_x = point[0]
    pt_y = point[1]
    cone_n = Polygon([pt_x - width_v // 2, pt_x + width_v // 2, pt_x],
                     [pt_y, pt_y, pt_y - height_v], 3)
    cone_n.translate(delta_x=0, delta_y=offset)
    cone_s = Polygon([pt_x - width_v // 2, pt_x + width_v // 2, pt_x],
                     [pt_y, pt_y, pt_y + height_v], 3)
    cone_s.translate(delta_x=0, delta_y=-offset)
    cone_e = Polygon([pt_x, pt_x, pt_x + height_h],
                     [pt_y + width_h // 2, pt_y - width_h // 2, pt_y], 3)
    cone_e.translate(delta_x=-offset, delta_y=0)
    cone_w = Polygon([pt_x, pt_x, pt_x - height_h],
                     [pt_y + width_h // 2, pt_y - width_h // 2, pt_y], 3)
    cone_w.translate(delta_x=offset, delta_y=0)
    return {'n': cone_n, 'e': cone_e, 's': cone_s, 'w': cone_w}


def sort_cluster_by_y_then_x(cluster, inverse_y=False, inverse_x=False):
    def y_then_x(a, b):
        if a[1][0][1] < b[1][0][1] or (a[1][0][1] == b[1][0][1] and a[1][0][0] < b[1][0][0]):
            return -1
        elif a[1][0] == b[1][0]:
            return 0
        else:
            return 1

    def y_then_ix(a, b):
        if a[1][0][1] < b[1][0][1] or (a[1][0][1] == b[1][0][1] and a[1][0][0] > b[1][0][0]):
            return -1
        elif a[1][0] == b[1][0]:
            return 0
        else:
            return 1

    def iy_then_x(a, b):
        if a[1][0][1] > b[1][0][1] or (a[1][0][1] == b[1][0][1] and a[1][0][0] < b[1][0][0]):
            return -1
        elif a[1][0] == b[1][0]:
            return 0
        else:
            return 1

    def iy_then_ix(a, b):
        if a[1][0][1] > b[1][0][1] or (a[1][0][1] == b[1][0][1] and a[1][0][0] > b[1][0][0]):
            return -1
        elif a[1][0] == b[1][0]:
            return 0
        else:
            return 1

    if inverse_y and inverse_x:
        cluster_sorted = sorted(cluster, key=functools.cmp_to_key(iy_then_ix))
    elif inverse_y:
        cluster_sorted = sorted(cluster, key=functools.cmp_to_key(iy_then_x))
    elif inverse_x:
        cluster_sorted = sorted(cluster, key=functools.cmp_to_key(y_then_ix))
    else:
        cluster_sorted = sorted(cluster, key=functools.cmp_to_key(y_then_x))

    return cluster_sorted


def check_horizontal_edge(point_a, point_b):
    # Bigger y-gap than x-gap
    if math.fabs(point_a[0] - point_b[0]) < math.fabs(point_a[1] - point_b[1]):
        is_horizontal = False
    # Bigger x-gap than y-gap
    else:
        is_horizontal = True
    return is_horizontal


def smooth_surrounding_polygon(polygon, poly_norm_dist=10, orientation_dims=(400, 800, 600, 400), offset=0):
    """
    Takes a "crooked" polygon and smooths it, by approximating vertical and horizontal edges.

    1.) The polygon gets normalized, where the resulting vertices are at most `poly_norm_dist` pixels apart.

    2.) For each vertex of the original polygon an orientation is determined:

    2.1) Four cones (North, East, South, West) are generated, with the dimensions given by `or_dims`
    (width_vertical, height_vertical, width_horizontal, height_horizontal), i.e. North and South rectangles
    have dimensions width_v x height_v, whereas East and West rectangles have dimensions width_h x height_h.

    2.2) The offset controls how far the cones overlap (e.g. how far the north cone gets translated south)

    2.3) Each rectangle counts the number of contained points from the normalized polygon

    2.4) The top two rectangle counts determine the orientation of the vertex: vertical, horizontal or one
    of the four possible corner types.

    3.) Vertices with a differing orientation to its agreeing neighbours are assumed to be mislabeled and
    get its orientation converted to its neighbours.

    4.) Corner clusters of the same type need to be shrunken down to one corner, with the rest being converted
    to verticals / horizontals.

    5.) Clusters between corners (corner-V/H-...-V/H-corner) get smoothed if they contain at least five points,
    by taking the average over the y-coordinates for horizontal edges and the average over the x-coordinates for
    vertical edges.

    :param polygon: (not necessarily closed) polygon, represented as a list of tuples (x,y)
    :param poly_norm_dist: int, distance between pixels in normalized polygon
    :param orientation_dims: tuple (width_v, height_v, width_h, height_h), the dimensions of the orientation rectangles
    :param offset: int, number of pixel that the orientation cones overlap
    :return: dict (keys = article_id, values = smoothed polygons)
    """
    if isinstance(polygon, Polygon):
        polygon = polygon.as_list()
    # Normalize polygon points over surrounding polygon
    surrounding_polygon = polygon.copy()
    # Optionally close polygon
    if surrounding_polygon[0] != surrounding_polygon[-1]:
        surrounding_polygon.append(polygon[0])

    # print("--------------------------")

    # Normalize polygon points
    poly_xs, poly_ys = zip(*surrounding_polygon)
    poly = Polygon(list(poly_xs), list(poly_ys), len(poly_xs))
    poly_norm = norm_poly_dists([poly], des_dist=poly_norm_dist)[0]

    # Get polygon dimensions
    poly_bb = poly.get_bounding_box()
    poly_h, poly_w = poly_bb.height, poly_bb.width
    # Build final dimensions for orientation objects
    dims_flex = [poly_h // 2, poly_h // 2, poly_w // 2, poly_h // 3]
    dims_min = [100, 80, 100, 60]
    dims = [max(min(x, y), z) for x, y, z in zip(orientation_dims, dims_flex, dims_min)]

    # print("poly_height {}, poly_width {}".format(poly_h, poly_w))
    # print("orientation_dims ", orientation_dims)
    # print("dims_flex ", dims_flex)
    # print("dims ", dims)

    # Determine orientation for every vertex of the (original) polygon
    oriented_points = []
    for pt in polygon:
        # Build up 4 cones in each direction (N, E, S, W)
        cones = get_orientation_cones(pt, dims, offset)
        # Count the number of contained points from the normalized polygon in each cone
        points_in_cones = {'n': 0, 'e': 0, 's': 0, 'w': 0}
        for o in cones:
            for pn in zip(poly_norm.x_points, poly_norm.y_points):
                if cones[o].contains_point(pn):
                    points_in_cones[o] += 1

        # Get orientation of vertex by top two counts
        sorted_counts = sorted(points_in_cones.items(), key=lambda kv: kv[1], reverse=True)
        top_two = list(zip(*sorted_counts))[0][:2]
        if 'n' in top_two and 's' in top_two:
            pt_o = 'vertical'
        elif 'e' in top_two and 'w' in top_two:
            pt_o = 'horizontal'
        elif 'e' in top_two and 's' in top_two:
            pt_o = 'corner_ul'
        elif 'w' in top_two and 's' in top_two:
            pt_o = 'corner_ur'
        elif 'w' in top_two and 'n' in top_two:
            pt_o = 'corner_dr'
        else:
            pt_o = 'corner_dl'
        # Append point and its orientation as a tuple
        oriented_points.append((pt, pt_o))

        # print("Type: {}, Counts: {}".format(pt_o, sorted_counts))

    # Fix wrongly classified points between two same classified ones
    for i in range(len(oriented_points)):
        if oriented_points[i - 1][1] != oriented_points[i][1] \
                and oriented_points[i - 1][1] == oriented_points[(i + 1) % len(oriented_points)][1] \
                and 'corner' not in oriented_points[i - 1][1]:
            oriented_points[i] = (oriented_points[i][0], oriented_points[i - 1][1])

    # Search for corner clusters of the same type and keep only one corner
    # TODO: Do we need to rearrange the list to start with a corner here already?
    # TODO: E.g. what if one of the clusters wraps around?
    for i in range(len(oriented_points)):
        # Found a corner
        if 'corner' in oriented_points[i][1]:
            # Get cluster (and IDs) with same corner type
            corner_cluster = [(i, oriented_points[i])]
            j = (i + 1) % len(oriented_points)
            while oriented_points[i][1] == oriented_points[j][1]:
                corner_cluster.append((j, oriented_points[j]))
                j = (j + 1) % len(oriented_points)
            if len(corner_cluster) > 1:
                # Keep corner based on type
                if 'ul' in oriented_points[i][1]:
                    cluster_sorted = sort_cluster_by_y_then_x(corner_cluster)
                elif 'ur' in oriented_points[i][1]:
                    cluster_sorted = sort_cluster_by_y_then_x(corner_cluster, inverse_x=True)
                elif 'dl' in oriented_points[i][1]:
                    cluster_sorted = sort_cluster_by_y_then_x(corner_cluster, inverse_y=True)
                else:
                    cluster_sorted = sort_cluster_by_y_then_x(corner_cluster, inverse_y=True, inverse_x=True)
                cluster_to_remove = cluster_sorted[1:]
                # Convert cluster to verticals (we don't care about the type of edge vertex later on)
                for c in cluster_to_remove:
                    idx = c[0]
                    oriented_points[idx] = (oriented_points[idx][0], 'vertical')

    # Rearrange oriented_points list to start with a corner and wrap it around
    corner_idx = 0
    for i, op in enumerate(oriented_points):
        if 'corner' in op[1]:
            # print("Rearrange corner at index", i)
            corner_idx = i
            break
    oriented_points = oriented_points[corner_idx:] + oriented_points[:corner_idx]
    oriented_points.append(oriented_points[0])

    # print("oriented points, ", oriented_points)

    # Go through the polygon and and get all corner IDs
    corner_ids = []
    for i, op in enumerate(oriented_points):
        if 'corner' in op[1]:
            corner_ids.append(i)

    # print("corner IDs, ", corner_ids)

    # Look at point clusters between neighboring corners
    # Build up list of alternating x- and y-coordinates (representing rays) and build up the polygon afterwards
    smoothed_edges = []

    # Check if we start with a horizontal edge
    # In this case, take the corresponding y-coordinate as the line/edge (otherwise x-coordinate)
    start_cluster = oriented_points[corner_ids[0]:corner_ids[1] + 1]
    # Look at corners, since this cluster will get approximated
    if len(start_cluster) > 3:
        is_horizontal = check_horizontal_edge(start_cluster[0][0], start_cluster[-1][0])
    # Look at first two points, since we take them as is
    else:
        is_horizontal = check_horizontal_edge(start_cluster[0][0], start_cluster[1][0])

    # j is the index for the x- or y-coordinate (horizontal = y, vertical = x)
    j = int(is_horizontal)

    # print("horizontal_edge_start", is_horizontal)

    for i in range(len(corner_ids) - 1):
        cluster = oriented_points[corner_ids[i]:corner_ids[i + 1] + 1]
        # Approximate edges with at least 4 points (including corners)
        if len(cluster) > 3:
            # Plausi-Check if we're getting the correct type of edge (between corners)
            # Else, switch it and insert missing ray beforehand
            if not j == check_horizontal_edge(cluster[0][0], cluster[-1][0]):
                # print("SWITCH", i)
                smoothed_edges.append(cluster[0][0][j])
                j = int(not j)

            mean = 0
            for pt in cluster:
                mean += pt[0][j]
            mean = round(float(mean) / len(cluster))
            smoothed_edges.append(mean)
            # Switch from x- to y-coordinate and vice versa
            j = int(not j)
        # Keep the rest as is, alternating between x- and y-coordinate for vertical / horizontal edges
        else:
            # Plausi-Check if we're getting the correct type of edge (between first two points)
            # Else, switch it and insert missing ray beforehand
            if not j == check_horizontal_edge(cluster[0][0], cluster[1][0]):
                # print("SWITCH", i)
                smoothed_edges.append(cluster[0][0][j])
                j = int(not j)

            # Exclude last point so we don't overlap in the next cluster
            for pt in cluster[:-1]:
                smoothed_edges.append(pt[0][j])
                j = int(not j)

        # print("smoothed_edges", smoothed_edges)

        # At last step, we may need to add another ray, if the edges between last & first don't match
        if i == len(corner_ids) - 2:
            if j != is_horizontal:
                smoothed_edges.append(cluster[-1][0][j])
                # print("smoothed_edges after last step\n", smoothed_edges)

    # Go over list of x-y values and build up the polygon by taking the intersection of the rays as vertices
    smoothed_polygon = Polygon()
    for i in range(len(smoothed_edges)):
        if is_horizontal:
            smoothed_polygon.add_point(smoothed_edges[(i + 1) % len(smoothed_edges)], smoothed_edges[i])
            is_horizontal = int(not is_horizontal)
        else:
            smoothed_polygon.add_point(smoothed_edges[i], smoothed_edges[(i + 1) % len(smoothed_edges)])
            is_horizontal = int(not is_horizontal)

    # print("polygon", smoothed_polygon)

    return smoothed_polygon


def bounding_box(points):
    """
    Computes the bounding box of a list of 2D points.

    :param points: list of (x, y) tuples, representing the points
    :return: list of four (x, y) tuples, representing the vertices of the bounding box
    """
    xs, ys = zip(*points)
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]


def convex_hull(points):
    """
    Computes the convex hull of a list of 2D points, by implementing Andrew's monotone chain algorithm.

    :param points: list of (x, y) tuples, representing the points
    :return: list of (x, y) tuples, representing the convex hull
    """

    def turn_left(p, q, r):
        """
        Returns `True` if the three points `p`, `q`, `r` constitute a 'left turn'.

        To do so, it computes the z-coordinate of the cross product of the two vectors (pq) & (pr).
        If the result is 0, the points are collinear. If it is positive, the three points constitute
        a 'left turn' (counter-clockwise), otherwise a 'right turn' (clockwise).
        """
        return (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]) > 0

    def x_then_y(a, b):
        if a[0] < b[0] or (a[0] == b[0] and a[1] < b[1]):
            return -1
        elif a == b:
            return 0
        else:
            return 1

    sorted_points = sorted(points, key=functools.cmp_to_key(x_then_y))

    # Build lower hull
    lower_hull = []
    for pt in sorted_points:
        while len(lower_hull) > 1 and not turn_left(lower_hull[-2], lower_hull[-1], pt):
            lower_hull.pop()
        lower_hull.append(pt)

    # Build upper hull
    upper_hull = []
    for pt in reversed(sorted_points):
        while len(upper_hull) > 1 and not turn_left(upper_hull[-2], upper_hull[-1], pt):
            upper_hull.pop()
        upper_hull.append(pt)

    return lower_hull[:-1] + upper_hull[:-1]


def alpha_shape(points, alpha):
    """
    Computation of the alpha-shape (concave hull) of a set of two dimensional points.
    Possible outliers might be ignored by the concave hull for too small values for alpha.

    :param points: np.array of shape (n,2) points
    :param alpha: alpha value > 0 (for alpha -> infinity the algorithm computes the convex hull)

    :return: sorted list of points [x,y] representing the edge vertices of the alpha-shape
    """

    def get_ordered_boundaries(edges):
        circle_edges, unvisited_edges = get_ordered_circles(edges=edges)
        circle_list = [circle_edges]

        while len(unvisited_edges) > 0:
            circle_edges, unvisited_edges = get_ordered_circles(edges=unvisited_edges)
            circle_list.append(circle_edges)

        return circle_list

    def get_ordered_circles(edges):
        if not edges:
            return [], []

        circle_edges = [edges[0]]
        unvisited_edges = edges[1:]

        while len(circle_edges) < len(edges):
            nothing = True

            for edge in edges:
                if edge in circle_edges or (edge[1], edge[0]) in circle_edges:
                    continue

                if edge[0] == circle_edges[-1][1]:
                    circle_edges.append(edge)
                    unvisited_edges.remove(edge)
                    nothing = False
                elif edge[1] == circle_edges[-1][1]:
                    circle_edges.append((edge[1], edge[0]))
                    unvisited_edges.remove(edge)
                    nothing = False

            if nothing:
                break

        return circle_edges, unvisited_edges

    assert alpha > 0, "alpha value has to be greater than zero"

    # algorithm needs at least four points
    if points.shape[0] <= 3:
        boundary_points = points.tolist()
        boundary_points.append(boundary_points[0])
        return boundary_points

    edges = []
    triangulation = Delaunay(points)

    # loop over all Delaunay triangles:
    # index_a, index_b, index_c = indices of the corner points of the triangle
    for index_a, index_b, index_c in triangulation.vertices:
        point_a = points[index_a]
        point_b = points[index_b]
        point_c = points[index_c]

        # computing radius of triangle circum circle:
        # lengths of sides of triangle
        a = np.linalg.norm(point_a - point_b)
        b = np.linalg.norm(point_b - point_c)
        c = np.linalg.norm(point_c - point_a)

        # semi perimeter of triangle
        s = (a + b + c) / 2.0

        # area of triangle by Heron's formula
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))

        # radius of triangle circum circle
        circum_r = a * b * c / (4.0 * (area + 1e-8))

        if circum_r < alpha:
            if (index_a, index_b) in edges:
                edges.remove((index_a, index_b))
            elif (index_b, index_a) in edges:
                edges.remove((index_b, index_a))
            else:
                edges.append((index_a, index_b))

            if (index_b, index_c) in edges:
                edges.remove((index_b, index_c))
            elif (index_c, index_b) in edges:
                edges.remove((index_c, index_b))
            else:
                edges.append((index_b, index_c))

            if (index_c, index_a) in edges:
                edges.remove((index_c, index_a))
            elif (index_a, index_c) in edges:
                edges.remove((index_a, index_c))
            else:
                edges.append((index_c, index_a))

    boundaries = get_ordered_boundaries(edges=edges)

    # no boundary edges or
    # boundary with several distant circles / several circles intersecting each other in one point
    if boundaries == [[]] or len(boundaries) > 1:
        print("alpha value not suitable -> is increased")
        return alpha_shape(points=points, alpha=alpha + alpha * 0.2)

    # boundary with several circles intersecting each other in one point
    edge_list = [j for i in boundaries[0] for j in i]
    edge_counter = Counter(edge_list)

    for edge in edge_counter:
        if edge_counter[edge] > 2:
            print("alpha value not suitable -> is increased")
            return alpha_shape(points, alpha=alpha + alpha * 0.2)

    edges = boundaries[0]
    boundary_points = []

    # get the coordinates of the ordered edge vertices
    for edge in edges:
        boundary_points.append(points[edge[0]].tolist())
    boundary_points.append(boundary_points[0])

    return boundary_points


def polygon_clip(poly, clip_poly):
    """
    Computes the intersection of an arbitray polygon with a convex clipping polygon, by
    implementing Sutherland-Hodgman's algorithm for polygon clipping.

    :param poly: list of tuples, representing the arbitrary polygon
    :param clip_poly: list of tuples, representing the convex clipping polygon, given in counter-clockwise order
    :return: list of tuples, representing the intersection / clipped polygon
    """

    def is_inside(r, e):
        """
        Returns `True` if the point `r` lies on the inside of the edge `e = (p,q)`.

        To do so, it computes the z-coordinate of the cross product of the two vectors `[pq]` & `[pr]`.
        If the result is 0, the points are collinear. If it is positive, the three points constitute
        a 'left turn' (counter-clockwise), otherwise a 'right turn' (clockwise).
        """
        p = e[0]
        q = e[1]
        return (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]) > 0

    def compute_intersection(e1, e2):
        """
        Computes the intersection point of the edges `e1` & `e2`.
        """
        # x-y-coordinates of the four points
        x1, y1 = e1[0][0], e1[0][1]
        x2, y2 = e1[1][0], e1[1][1]
        x3, y3 = e2[0][0], e2[0][1]
        x4, y4 = e2[1][0], e2[1][1]
        # help variables to reduce computations
        dx12 = x1 - x2
        dx34 = x3 - x4
        dy12 = y1 - y2
        dy34 = y3 - y4
        # nominator part 1 and 2
        n1 = x1 * y2 - y1 * x2
        n2 = x3 * y4 - y3 * x4
        # denominator
        d = 1.0 / (dx12 * dy34 - dy12 * dx34)
        # intersection point
        return (n1 * dx34 - dx12 * n2) * d, (n1 * dy34 - dy12 * n2) * d

    output_poly = poly
    c1 = clip_poly[-1]
    # clip poly against each edge in clip_poly
    for c2 in clip_poly:
        # input is the clipped output from the previous run
        input_poly = output_poly
        # we build the new output from scratch
        output_poly = []
        clip_edge = (c1, c2)
        p1 = input_poly[-1]
        # go over each poly edge individually
        for p2 in input_poly:
            poly_edge = (p1, p2)
            # add (and implicitly remove) points depending on the four cases
            if is_inside(p2, clip_edge):
                if not is_inside(p1, clip_edge):
                    output_poly.append(compute_intersection(poly_edge, clip_edge))
                output_poly.append(p2)
            elif is_inside(p1, clip_edge):
                output_poly.append(compute_intersection(poly_edge, clip_edge))
            # go to next poly edge
            p1 = p2
        # no intersection
        if not output_poly:
            return []
        # go to next clip edge
        c1 = c2

    return output_poly


def get_dist_fast(point, bb):
    """ Calculate the distance between a ``point`` and a bounding box ``bb`` by adding up the x- and y-distance.

    :param point: a point given by [x, y]
    :param bb: the bounding box of a baseline polygon
    :type point: list of float
    :type bb: Rectangle
    :return: the distance of the point to the bounding box
    """
    dist = 0.0

    if point[0] < bb.x:
        dist += bb.x - point[0]
    if point[0] > bb.x + bb.width:
        dist += point[0] - bb.x - bb.width
    if point[1] < bb.y:
        dist += bb.y - point[1]
    if point[1] > bb.y + bb.height:
        dist += point[1] - bb.y - bb.height

    return dist


def get_in_dist(p1, p2, or_vec_x, or_vec_y):
    """ Calculate the inline distance of the points ``p1`` and ``p2`` according to the orientation vector with
    x-coordinate ``or_vec_x`` and y-coordinate ``or_vec_y``.

    :param p1: first point
    :param p2: second point
    :param or_vec_x: x-coordinate of the orientation vector
    :param or_vec_y: y-coordinate of the orientation vector
    :return: the inline distance of the points p1 and p2 according to the given orientation vector
    """
    diff_x = p1[0] - p2[0]
    diff_y = -p1[1] + p2[1]

    # Parallel component of (diff_x, diff_y) is lambda * (or_vec_x, or_vec_y) with lambda:
    return diff_x * or_vec_x + diff_y * or_vec_y


def get_off_dist(p1, p2, or_vec_x, or_vec_y):
    """ Calculate the offline distance of the points ``p1`` and ``p2`` according to the orientation vector with
    x-coordinate ``or_vec_x`` and y-coordinate ``or_vec_y``.

    :param p1: first point
    :param p2: second point
    :param or_vec_x: x-coordinate of the orientation vector
    :param or_vec_y: y-coordinate of the orientation vector
    :return: the offline distance of the points p1 and p2 according to the given orientation vector
    """
    diff_x = p1[0] - p2[0]
    diff_y = -p1[1] + p2[1]

    return diff_x * or_vec_y - diff_y * or_vec_x


def calc_tols(polys_truth, tick_dist=5, max_d=250, rel_tol=0.25):
    """ Calculate tolerance values for every GT baseline according to https://arxiv.org/pdf/1705.03311.pdf.

    :param polys_truth: groundtruth baseline polygons (normalized)
    :param tick_dist: desired distance of points of the baseline polygon (default: 5)
    :param max_d: max distance of pixels of a baseline polygon to any other baseline polygon (distance in terms of the
    x- and y-distance of the point to a bounding box of another polygon - see get_dist_fast) (default: 250)
    :param rel_tol: relative tolerance value (default: 0.25)
    :type polys_truth: list of Polygon
    :return: tolerance values of the GT baselines
    """
    tols = []

    for poly_a in polys_truth:
        # Calculate the angle of the linear regression line representing the baseline polygon poly_a
        angle = calc_reg_line_stats(poly_a)[0]
        # Orientation vector (given by angle) of length 1
        or_vec_y, or_vec_x = math.sin(angle), math.cos(angle)
        dist = max_d
        # first and last point of polygon
        pt_a1 = [poly_a.x_points[0], poly_a.y_points[0]]
        pt_a2 = [poly_a.x_points[-1], poly_a.y_points[-1]]

        # iterate over pixels of the current GT baseline polygon
        for x_a, y_a in zip(poly_a.x_points, poly_a.y_points):
            p_a = [x_a, y_a]
            # iterate over all other polygons (to calculate X_G)
            for poly_b in polys_truth:
                if poly_b != poly_a:
                    # if polygon poly_b is too far away from pixel p_a, skip
                    if get_dist_fast(p_a, poly_b.get_bounding_box()) > dist:
                        continue

                    # get first and last pixel of baseline polygon poly_b
                    pt_b1 = poly_b.x_points[0], poly_b.y_points[0]
                    pt_b2 = poly_b.x_points[-1], poly_b.y_points[-1]

                    # calculate the inline distance of the points
                    in_dist1 = get_in_dist(pt_a1, pt_b1, or_vec_x, or_vec_y)
                    in_dist2 = get_in_dist(pt_a1, pt_b2, or_vec_x, or_vec_y)
                    in_dist3 = get_in_dist(pt_a2, pt_b1, or_vec_x, or_vec_y)
                    in_dist4 = get_in_dist(pt_a2, pt_b2, or_vec_x, or_vec_y)
                    if (in_dist1 < 0 and in_dist2 < 0 and in_dist3 < 0 and in_dist4 < 0) or (
                            in_dist1 > 0 and in_dist2 > 0 and in_dist3 > 0 and in_dist4 > 0):
                        continue

                    for p_b in zip(poly_b.x_points, poly_b.y_points):
                        if abs(get_in_dist(p_a, p_b, or_vec_x, or_vec_y)) <= 2 * tick_dist:
                            dist = min(dist, abs(get_off_dist(p_a, p_b, or_vec_x, or_vec_y)))
        if dist < max_d:
            tols.append(dist)
        else:
            tols.append(0)

    sum_tols = 0.0
    num_tols = 0
    for tol in tols:
        if tol != 0:
            sum_tols += tol
            num_tols += 1

    mean_tols = max_d
    if num_tols:
        mean_tols = sum_tols / num_tols

    for i, tol in enumerate(tols):
        if tol == 0:
            tols[i] = mean_tols
        tols[i] = min(tols[i], mean_tols)
        tols[i] *= rel_tol

    return tols
