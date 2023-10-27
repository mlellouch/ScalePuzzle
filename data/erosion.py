import cv2
import numpy as np
from random import uniform
import math
import itertools
from matplotlib import path as mplp
from skimage.measure import find_contours


# Noise Generation Methods

def calculate_epsilon(polys, noise_volume, noise_reference='diameter'):
    """
    Calculates the noise absolute value (epsilon) for a given noise volume and a list of polygons. The defaultive noise reference is the shape's diameter.
    :param polys: The shape of the puzzle.
    :param noise_volume: The volume of noise to be applied (relative to the reference).
    :param noise_reference: The reference to be used for calculating the noise volume. Can be 'diameter', 'average_edge_length' or 'min_edge_length'.
    :param is_relative: If True, the noise would be relative for each polygon. If False, the noise volume would be calculated for the whole puzzle.
    """
    if noise_volume <= 0:
        return 0
    if noise_reference == 'diameter':
        reference = calculate_diameter(polys)
    elif noise_reference == 'average_edge_length':
        reference = calculate_average_edge_length(polys)
    elif noise_reference == 'min_edge_length':
        reference = calculate_min_edge_length(polys)
    else:
        raise ValueError(
            'Invalid noise reference. Must be one of "diameter", "average_edge_length" or "min_edge_length".')
    return (noise_volume / 100) * reference


def calculate_global_shape(pieces):
    to_convex = np.array(list(itertools.chain.from_iterable(pieces)))
    return cv2.convexHull(to_convex.astype(np.float32)).squeeze()


def calculate_diameter(pieces):
    """
    Calculates the diameter of a given set of polygons (can also contain a single polygon), by calculating their convex hull.
    """
    if len(pieces) > 1:
        shape = calculate_global_shape(pieces)
    else:
        shape = pieces[0]
    diameter = 0
    for i, p1 in enumerate(shape):
        for p2 in shape[i + 1:]:
            length = np.linalg.norm([p2[0] - p1[0], p2[1] - p1[1]])
            if length > diameter:
                diameter = length
    return diameter


def calculate_average_edge_length(pieces):
    edge_sizes_sum = 0
    for piece in pieces:
        avg_edge = 0
        for i, v in enumerate(piece):
            if i == 0:
                avg_edge = np.linalg.norm([piece[-1][0] - v[0], piece[-1][1] - v[1]])
            else:
                avg_edge += np.linalg.norm([piece[i - 1][0] - v[0], piece[i - 1][1] - v[1]])
        edge_sizes_sum += avg_edge / len(piece)
    return edge_sizes_sum / len(pieces)


def calculate_min_edge_length(pieces):
    min_edge = np.inf
    for piece in pieces:
        for i, v in enumerate(piece):
            if i == 0:
                edge = np.linalg.norm([piece[-1][0] - v[0], piece[-1][1] - v[1]])
            else:
                edge = np.linalg.norm([piece[i - 1][0] - v[0], piece[i - 1][1] - v[1]])
            if edge < min_edge:
                min_edge = edge
    return min_edge


# Given a puzzle and epsilon (and an optional threshold) , returns a new puzzle with noised pieces and fixed
# relationships while safely cleaning automatically-erased pieces (and also pieces whose diameter is smaller than
# threshold, if given one)
# def apply_noise_on_puzzle(pieces, rels, noise_volume=0, is_relative=False, noise_reference='diameter', threshold=-1, fixed_noise_value=-1, prevent_erased_pieces=False):
#     """
#     Given a puzzle, noise parameters and an optional threshold, returns a new puzzle with noised pieces and fixed pieces' relationships while safely cleaning
#     automatically-erased pieces (and also pieces whose diameter is smaller than threshold, if given one).
#     Returns the new puzzle, an indices map for the erased pieces, the indices of the erased pieces and the absolute noise value (epsilon).
#     :param pieces: The puzzle's pieces.
#     :param rels: The pieces' relationships.
#     :param noise_volume: The volume of noise to be applied (relative to the reference).
#     :param is_relative: If True, the noise would be relative for each piece. If False, the noise volume would be calculated for the whole puzzle.
#     :param noise_reference: The reference to be used for calculating the noise volume. Can be 'diameter', 'average_edge_length' or 'min_edge_length'.
#     :param threshold: The threshold for the pieces' diameters. If a piece's diameter is smaller than the threshold, it would be erased.
#     :param fixed_noise_value: An optional fixed noise value to be applied on the pieces. If given, it would be used as the absolute noise value.
#     :param prevent_erased_pieces: If True, no pieces would be erased by applying noise on them, which would instead decrease untill they wouldn't be erased.
#     """
#     noised_pieces, erased_indices, epsilon, epsilons_dict = apply_noise_on_pieces(pieces, is_relative, noise_volume, noise_reference, threshold, fixed_noise_value, prevent_erased_pieces)
#     new_puzzle, indices_map = delete_pieces_from_existing_puzzle(noised_pieces, rels, erased_indices)

#     return new_puzzle, indices_map, erased_indices, epsilon, epsilons_dict


# For given epsilon and pieces, applies noise on the pieces and documents automatically erased pieces (returns them
# as empty lists). Can also check if a noised piece's diameter is smaller than a pre determined threshold,
# and erase it too.
def apply_noise_on_pieces(pieces, is_relative=False, noise_volume=0, noise_reference='diameter', threshold=-1,
                          fixed_noise_value=-1, prevent_erased_pieces=False):
    """
    For a given set of pieces and a noise volume, calculates an absolute noise value for each piece,
    applies it on the pieces and documents automatically erased pieces (returns them as empty lists).
    Can also check if a noised piece's diameter is smaller than a pre determined threshold, and erase it too.
    Returns the noised pieces,  the indexes of the erased pieces, and the average/global absolute noise value (epsilon).
    :param pieces: The pieces to be noised.
    :param fixed_epsilon: If given, the noise value to be applied on all pieces. If not given, the noise value would be calculated.
    :param threshold: If given, the minimum diameter a piece can have. If a piece's diameter is smaller than the threshold, it would be erased.
    :param is_relative: If True, the noise would be relative for each piece. If False, the noise volume would be calculated for the whole puzzle.
    :param noise_volume: The volume of noise to be applied (relative to the reference).
    :param noise_reference: The reference to be used for calculating the noise value. Can be 'diameter', 'average_edge_length' or 'min_edge_length'.
    :param prevent_erased_pieces: If True, no pieces would be erased by applying noise on them, which would instead decrease untill they wouldn't be erased.
    """
    noised_pieces = []
    erased_pieces_indexes = []
    epsilons = []
    epsilons_dict = {}
    if noise_volume <= 0.0 and fixed_noise_value <= 0:
        return pieces, erased_pieces_indexes, 0, epsilons_dict
    if not is_relative:
        if fixed_noise_value == -1:
            epsilon = calculate_epsilon(pieces, noise_volume, noise_reference)
        else:
            epsilon = fixed_noise_value
    for i, piece in enumerate(pieces):
        if is_relative:
            epsilon = calculate_epsilon([piece], noise_volume, noise_reference)
            epsilons.append(epsilon)
        noised_piece, erased = apply_noise_on_piece(piece, epsilon)
        if erased:
            if prevent_erased_pieces:
                reduced_epsilon = epsilon
                while erased:
                    reduced_epsilon = reduced_epsilon / 2
                    noised_piece, erased = apply_noise_on_piece(piece, reduced_epsilon)
                noised_pieces.append(noised_piece)
                epsilons_dict[i] = reduced_epsilon
            else:
                erased_pieces_indexes.append(i)
                noised_pieces.append([])
        elif threshold != -1:
            if is_piece_too_small(noised_piece, threshold):
                erased_pieces_indexes.append(i)
                noised_pieces.append([])
            else:
                noised_pieces.append(noised_piece)
        else:
            noised_pieces.append(noised_piece)
            epsilons_dict[i] = epsilon
    if is_relative:
        epsilon = np.mean(epsilons)
    return noised_pieces, erased_pieces_indexes, epsilon, epsilons_dict


# Calculates a piece's diameter and checks if its smaller than a pre-determined threshold.
def is_piece_too_small(piece, threshold):
    return calculate_diameter([piece]) < threshold


def are_points_in_poly(points, poly):
    poly_p = mplp.Path([(p[0], p[1]) for p in poly])
    points_as_tuples = [(p[0], p[1]) for p in points]
    contained = poly_p.contains_points(points_as_tuples)
    return contained.all()


# For every vertice in poly:
# 1. Generates a unit vector, directed to a random directionl between its two edges.
# 2. Randomly generates the noise size (to a limit of epsilon), multiplies it by the direction vector
# and adds it to the vertice (thus moving it).
# 3. Checks if the new noised poly intersects with itself, or was dragged out of the original poly's borders
# (thus, got "erased" due to the noise)
def apply_noise_on_piece(poly, epsilon):
    noised_poly = []
    erased = False
    original_diameter = calculate_diameter([poly])

    for i, v in enumerate(poly):
        if i == 0:
            neighbors = [poly[i + 1], poly[-1]]
        elif i == len(poly) - 1:
            neighbors = [poly[0], poly[i - 1]]
        else:
            neighbors = [poly[i - 1], poly[i + 1]]
        direction = get_direction_angle(v, neighbors)
        noise_size = uniform(0, epsilon)
        if noise_size >= original_diameter:
            erased = True
        direction_unit = np.array([math.cos(np.radians(direction)), math.sin(np.radians(direction))])
        noised_v = v + noise_size * direction_unit
        if not are_points_in_poly([noised_v], poly):
            erased = True
        noised_poly.append(noised_v)
    if not erased:
        erased = check_self_intersection(noised_poly)
    if erased:
        return noised_poly, erased
    return np.array(noised_poly), erased


def relative_apply_noise_on_piece(poly, epsilon=0, is_relative=False, noise_volume=0, noise_reference='diameter'):
    """
    :param poly: the polygon to be noised
    :param epsilon: a fixed pre-determined noise size, if such exists
    :param is_relative: a boolean value, representing wether the noise size is relative to the piece's size
    :param noise_volume: the noise volume, relative to the noise_reference
    :param noise_reference: the reference for the noise volume. Can be 'diameter', 'average_edge_length' or 'min_edge_length'
    """
    noised_poly = []
    erased = False
    original_diameter = calculate_diameter([poly])
    if is_relative:
        epsilon = calculate_epsilon(poly, noise_volume, noise_reference)

    for i, v in enumerate(poly):
        if i == 0:
            neighbors = [poly[i + 1], poly[-1]]
        elif i == len(poly) - 1:
            neighbors = [poly[0], poly[i - 1]]
        else:
            neighbors = [poly[i - 1], poly[i + 1]]
        direction = get_direction_angle(v, neighbors)
        noise_size = uniform(0, epsilon)
        if noise_size >= original_diameter:
            erased = True
        direction_unit = np.array([math.cos(np.radians(direction)), math.sin(np.radians(direction))])
        noised_v = v + noise_size * direction_unit
        if not are_points_in_poly([noised_v], poly):
            erased = True
        noised_poly.append(noised_v)
    if not erased:
        erased = check_self_intersection(noised_poly)
    if erased:
        return noised_poly, erased
    return np.array(noised_poly), erased


def get_direction_angle(base_pt, neighbors):
    line1 = [neighbors[0][0] - base_pt[0], neighbors[0][1] - base_pt[1]]
    line2 = [neighbors[1][0] - base_pt[0], neighbors[1][1] - base_pt[1]]
    orientation = check_orientation(neighbors[0], base_pt, neighbors[1])
    if (orientation == -1 and line1[1] > line2[1]) or (orientation == 1 and line1[1] < line2[1]):
        angle1 = np.degrees(np.arctan2(line1[1], line1[0]))
        angle2 = np.degrees(np.arctan2(line2[1], line2[0]))
    else:
        angle1 = (np.degrees(np.arctan2(line1[1], line1[0])) + 360) % 360
        angle2 = (np.degrees(np.arctan2(line2[1], line2[0])) + 360) % 360
    return np.random.uniform(min(angle1, angle2), max(angle1, angle2))
    # return (uniform(min(angle1, angle2), max(angle1, angle2)) + 2 * np.pi) % (2*np.pi)


def polysides(polygon):
    """
    Returns the sides of the polygon
    described as line segments.

    e.g

    For the polygon [p1, p2, p3], the result will be
    [[p1, p2], [p2, p3], [p3, p1]
    """
    return np.c_[polygon, np.roll(polygon, -1, axis=0)]


# Checks if a polygon is self-intersecting, by checking if one pair of its non-neighboring edges intersects
def check_self_intersection(poly):
    edges = polysides(poly)
    for i, edge1 in enumerate(edges):
        for edge2 in edges[i + 2:-1]:
            if check_edges_intersection(edge1, edge2):
                return True
        if (i != 0) and (i < len(edges) - 2):
            if check_edges_intersection(edge1, edges[-1]):
                return True
    return False


# Checks if 2 edges (of type [x1 y1 x2 y2]) intersect by comparing points triplets' orientation and covering general
# and special cases
def check_edges_intersection(edge1, edge2):
    p1 = [edge1[0], edge1[1]]
    p2 = [edge1[2], edge1[3]]
    p3 = [edge2[0], edge2[1]]
    p4 = [edge2[2], edge2[3]]

    orientation1_2_3 = check_orientation(p1, p2, p3)
    if (orientation1_2_3 == 0) and on_segment(p1, p2, p3):  # Checks special case
        return True
    orientation1_2_4 = check_orientation(p1, p2, p4)
    if (orientation1_2_4 == 0) and on_segment(p1, p2, p4):  # Checks special case
        return True
    orientation3_4_1 = check_orientation(p3, p4, p1)
    if (orientation3_4_1 == 0) and on_segment(p3, p4, p1):  # Checks special case
        return True
    orientation3_4_2 = check_orientation(p3, p4, p2)
    if (orientation3_4_2 == 0) and on_segment(p3, p4, p2):  # Checks special case
        return True

    # General cases
    if (orientation1_2_3 != orientation1_2_4) and (orientation3_4_1 != orientation3_4_2):
        return True

    return False


# Checks the orientation between 3 given points (by comparing each pair of points' slope). Returns 1 for clockwise,
# 0 for collinear, -1 for counter-clockwise
def check_orientation(p1, p2, p3):
    val = (float(p2[1] - p1[1]) * float(p3[0] - p2[0])) - (float(p2[0] - p1[0]) * float(p3[1] - p2[1]))
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0


# Given 3 collinear points, checks if p3 is on the line segment [p1,p2]
def on_segment(p1, p2, p3):
    if ((p3[0] <= max(p2[0], p1[0])) and (p3[0] >= min(p2[0], p1[0])) and (p3[1] <= max(p2[1], p1[1])) and (
            p3[1] >= min(p2[1], p1[1]))):
        return True
    return False


def erode_image(image, noise_volume=0, erosion_type="reference", noise_reference="average_edge_length"):
    """
    Erodes an image by applying noise on it.
    :param image: The image to be eroded.
    :param noise_volume: The volume of noise to be applied.
    :param erosion_type: The type of erosion to be applied. Can be 'reference' or 'sandstorm'.
    :param noise_reference: The reference to be used for calculating the noise volume, in case of 'reference' erosion type. Can be 'diameter', 'average_edge_length' or 'min_edge_length'.
    """

    if erosion_type == "reference":
        contours = find_contours(image[:, :, 3], 0)
        contours = [contour[:, ::-1] for contour in contours]
        if len(contours) == 0:
            contours = [np.array([[0, 0], [0, image.shape[0]], [image.shape[1], image.shape[0]], [image.shape[1], 0]])]
        # if reference == "diameter":
        #     noise_reference = calculate_diameter(contours)
        # elif reference == "average_edge":
        #     noise_reference = calculate_average_edge_length(contours)
        # elif reference == "min_edge":
        #     noise_reference = calculate_min_edge_length(contours)
        noise = calculate_epsilon(contours, noise_volume, noise_reference)
        eroded_contours, erased = apply_noise_on_piece(contours[0], noise)
        if not erased:
            eroded_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(eroded_mask, pts=[eroded_contours.astype(np.int32)], color=(255))
            eroded_image = np.zeros(image.shape, dtype=np.uint8)
            eroded_image[:, :, 3] = eroded_mask
            eroded_image[:, :, :3] = image[:, :, :3]
            return eroded_image

    elif erosion_type == "sandstorm":
        return "Not implemented yet"
    else:
        raise ValueError('Invalid erosion type. Must be one of "reference" or "sandstorm".')
        import cv2


import numpy as np
from random import uniform
import math
import itertools
from matplotlib import path as mplp
from skimage.measure import find_contours


# Noise Generation Methods

def calculate_epsilon(polys, noise_volume, noise_reference='diameter'):
    """
    Calculates the noise absolute value (epsilon) for a given noise volume and a list of polygons. The defaultive noise reference is the shape's diameter.
    :param polys: The shape of the puzzle.
    :param noise_volume: The volume of noise to be applied (relative to the reference).
    :param noise_reference: The reference to be used for calculating the noise volume. Can be 'diameter', 'average_edge_length' or 'min_edge_length'.
    :param is_relative: If True, the noise would be relative for each polygon. If False, the noise volume would be calculated for the whole puzzle.
    """
    if noise_volume <= 0:
        return 0
    if noise_reference == 'diameter':
        reference = calculate_diameter(polys)
    elif noise_reference == 'average_edge_length':
        reference = calculate_average_edge_length(polys)
    elif noise_reference == 'min_edge_length':
        reference = calculate_min_edge_length(polys)
    else:
        raise ValueError(
            'Invalid noise reference. Must be one of "diameter", "average_edge_length" or "min_edge_length".')
    return (noise_volume / 100) * reference


def calculate_global_shape(pieces):
    to_convex = np.array(list(itertools.chain.from_iterable(pieces)))
    return cv2.convexHull(to_convex.astype(np.float32)).squeeze()


def calculate_diameter(pieces):
    """
    Calculates the diameter of a given set of polygons (can also contain a single polygon), by calculating their convex hull.
    """
    if len(pieces) > 1:
        shape = calculate_global_shape(pieces)
    else:
        shape = pieces[0]
    diameter = 0
    for i, p1 in enumerate(shape):
        for p2 in shape[i + 1:]:
            length = np.linalg.norm([p2[0] - p1[0], p2[1] - p1[1]])
            if length > diameter:
                diameter = length
    return diameter


def calculate_average_edge_length(pieces):
    edge_sizes_sum = 0
    for piece in pieces:
        avg_edge = 0
        for i, v in enumerate(piece):
            if i == 0:
                avg_edge = np.linalg.norm([piece[-1][0] - v[0], piece[-1][1] - v[1]])
            else:
                avg_edge += np.linalg.norm([piece[i - 1][0] - v[0], piece[i - 1][1] - v[1]])
        edge_sizes_sum += avg_edge / len(piece)
    return edge_sizes_sum / len(pieces)


def calculate_min_edge_length(pieces):
    min_edge = np.inf
    for piece in pieces:
        for i, v in enumerate(piece):
            if i == 0:
                edge = np.linalg.norm([piece[-1][0] - v[0], piece[-1][1] - v[1]])
            else:
                edge = np.linalg.norm([piece[i - 1][0] - v[0], piece[i - 1][1] - v[1]])
            if edge < min_edge:
                min_edge = edge
    return min_edge


# Given a puzzle and epsilon (and an optional threshold) , returns a new puzzle with noised pieces and fixed
# relationships while safely cleaning automatically-erased pieces (and also pieces whose diameter is smaller than
# threshold, if given one)
# def apply_noise_on_puzzle(pieces, rels, noise_volume=0, is_relative=False, noise_reference='diameter', threshold=-1, fixed_noise_value=-1, prevent_erased_pieces=False):
#     """
#     Given a puzzle, noise parameters and an optional threshold, returns a new puzzle with noised pieces and fixed pieces' relationships while safely cleaning
#     automatically-erased pieces (and also pieces whose diameter is smaller than threshold, if given one).
#     Returns the new puzzle, an indices map for the erased pieces, the indices of the erased pieces and the absolute noise value (epsilon).
#     :param pieces: The puzzle's pieces.
#     :param rels: The pieces' relationships.
#     :param noise_volume: The volume of noise to be applied (relative to the reference).
#     :param is_relative: If True, the noise would be relative for each piece. If False, the noise volume would be calculated for the whole puzzle.
#     :param noise_reference: The reference to be used for calculating the noise volume. Can be 'diameter', 'average_edge_length' or 'min_edge_length'.
#     :param threshold: The threshold for the pieces' diameters. If a piece's diameter is smaller than the threshold, it would be erased.
#     :param fixed_noise_value: An optional fixed noise value to be applied on the pieces. If given, it would be used as the absolute noise value.
#     :param prevent_erased_pieces: If True, no pieces would be erased by applying noise on them, which would instead decrease untill they wouldn't be erased.
#     """
#     noised_pieces, erased_indices, epsilon, epsilons_dict = apply_noise_on_pieces(pieces, is_relative, noise_volume, noise_reference, threshold, fixed_noise_value, prevent_erased_pieces)
#     new_puzzle, indices_map = delete_pieces_from_existing_puzzle(noised_pieces, rels, erased_indices)

#     return new_puzzle, indices_map, erased_indices, epsilon, epsilons_dict


# For given epsilon and pieces, applies noise on the pieces and documents automatically erased pieces (returns them
# as empty lists). Can also check if a noised piece's diameter is smaller than a pre determined threshold,
# and erase it too.
def apply_noise_on_pieces(pieces, is_relative=False, noise_volume=0, noise_reference='diameter', threshold=-1,
                          fixed_noise_value=-1, prevent_erased_pieces=False):
    """
    For a given set of pieces and a noise volume, calculates an absolute noise value for each piece,
    applies it on the pieces and documents automatically erased pieces (returns them as empty lists).
    Can also check if a noised piece's diameter is smaller than a pre determined threshold, and erase it too.
    Returns the noised pieces,  the indexes of the erased pieces, and the average/global absolute noise value (epsilon).
    :param pieces: The pieces to be noised.
    :param fixed_epsilon: If given, the noise value to be applied on all pieces. If not given, the noise value would be calculated.
    :param threshold: If given, the minimum diameter a piece can have. If a piece's diameter is smaller than the threshold, it would be erased.
    :param is_relative: If True, the noise would be relative for each piece. If False, the noise volume would be calculated for the whole puzzle.
    :param noise_volume: The volume of noise to be applied (relative to the reference).
    :param noise_reference: The reference to be used for calculating the noise value. Can be 'diameter', 'average_edge_length' or 'min_edge_length'.
    :param prevent_erased_pieces: If True, no pieces would be erased by applying noise on them, which would instead decrease untill they wouldn't be erased.
    """
    noised_pieces = []
    erased_pieces_indexes = []
    epsilons = []
    epsilons_dict = {}
    if noise_volume <= 0.0 and fixed_noise_value <= 0:
        return pieces, erased_pieces_indexes, 0, epsilons_dict
    if not is_relative:
        if fixed_noise_value == -1:
            epsilon = calculate_epsilon(pieces, noise_volume, noise_reference)
        else:
            epsilon = fixed_noise_value
    for i, piece in enumerate(pieces):
        if is_relative:
            epsilon = calculate_epsilon([piece], noise_volume, noise_reference)
            epsilons.append(epsilon)
        noised_piece, erased = apply_noise_on_piece(piece, epsilon)
        if erased:
            if prevent_erased_pieces:
                reduced_epsilon = epsilon
                while erased:
                    reduced_epsilon = reduced_epsilon / 2
                    noised_piece, erased = apply_noise_on_piece(piece, reduced_epsilon)
                noised_pieces.append(noised_piece)
                epsilons_dict[i] = reduced_epsilon
            else:
                erased_pieces_indexes.append(i)
                noised_pieces.append([])
        elif threshold != -1:
            if is_piece_too_small(noised_piece, threshold):
                erased_pieces_indexes.append(i)
                noised_pieces.append([])
            else:
                noised_pieces.append(noised_piece)
        else:
            noised_pieces.append(noised_piece)
            epsilons_dict[i] = epsilon
    if is_relative:
        epsilon = np.mean(epsilons)
    return noised_pieces, erased_pieces_indexes, epsilon, epsilons_dict


# Calculates a piece's diameter and checks if its smaller than a pre-determined threshold.
def is_piece_too_small(piece, threshold):
    return calculate_diameter([piece]) < threshold


def are_points_in_poly(points, poly):
    poly_p = mplp.Path([(p[0], p[1]) for p in poly])
    points_as_tuples = [(p[0], p[1]) for p in points]
    contained = poly_p.contains_points(points_as_tuples)
    return contained.all()


# For every vertice in poly:
# 1. Generates a unit vector, directed to a random directionl between its two edges.
# 2. Randomly generates the noise size (to a limit of epsilon), multiplies it by the direction vector
# and adds it to the vertice (thus moving it).
# 3. Checks if the new noised poly intersects with itself, or was dragged out of the original poly's borders
# (thus, got "erased" due to the noise)
def apply_noise_on_piece(poly, epsilon):
    noised_poly = []
    erased = False
    original_diameter = calculate_diameter([poly])

    for i, v in enumerate(poly):
        if i == 0:
            neighbors = [poly[i + 1], poly[-1]]
        elif i == len(poly) - 1:
            neighbors = [poly[0], poly[i - 1]]
        else:
            neighbors = [poly[i - 1], poly[i + 1]]
        direction = get_direction_angle(v, neighbors)
        noise_size = uniform(0, epsilon)
        if noise_size >= original_diameter:
            erased = True
        direction_unit = np.array([math.cos(np.radians(direction)), math.sin(np.radians(direction))])
        noised_v = v + noise_size * direction_unit
        if not are_points_in_poly([noised_v], poly):
            erased = True
        noised_poly.append(noised_v)
    if not erased:
        erased = check_self_intersection(noised_poly)
    if erased:
        return noised_poly, erased
    return np.array(noised_poly), erased


def relative_apply_noise_on_piece(poly, epsilon=0, is_relative=False, noise_volume=0, noise_reference='diameter'):
    """
    :param poly: the polygon to be noised
    :param epsilon: a fixed pre-determined noise size, if such exists
    :param is_relative: a boolean value, representing wether the noise size is relative to the piece's size
    :param noise_volume: the noise volume, relative to the noise_reference
    :param noise_reference: the reference for the noise volume. Can be 'diameter', 'average_edge_length' or 'min_edge_length'
    """
    noised_poly = []
    erased = False
    original_diameter = calculate_diameter([poly])
    if is_relative:
        epsilon = calculate_epsilon(poly, noise_volume, noise_reference)

    for i, v in enumerate(poly):
        if i == 0:
            neighbors = [poly[i + 1], poly[-1]]
        elif i == len(poly) - 1:
            neighbors = [poly[0], poly[i - 1]]
        else:
            neighbors = [poly[i - 1], poly[i + 1]]
        direction = get_direction_angle(v, neighbors)
        noise_size = uniform(0, epsilon)
        if noise_size >= original_diameter:
            erased = True
        direction_unit = np.array([math.cos(np.radians(direction)), math.sin(np.radians(direction))])
        noised_v = v + noise_size * direction_unit
        if not are_points_in_poly([noised_v], poly):
            erased = True
        noised_poly.append(noised_v)
    if not erased:
        erased = check_self_intersection(noised_poly)
    if erased:
        return noised_poly, erased
    return np.array(noised_poly), erased


def get_direction_angle(base_pt, neighbors):
    line1 = [neighbors[0][0] - base_pt[0], neighbors[0][1] - base_pt[1]]
    line2 = [neighbors[1][0] - base_pt[0], neighbors[1][1] - base_pt[1]]
    orientation = check_orientation(neighbors[0], base_pt, neighbors[1])
    if (orientation == -1 and line1[1] > line2[1]) or (orientation == 1 and line1[1] < line2[1]):
        angle1 = np.degrees(np.arctan2(line1[1], line1[0]))
        angle2 = np.degrees(np.arctan2(line2[1], line2[0]))
    else:
        angle1 = (np.degrees(np.arctan2(line1[1], line1[0])) + 360) % 360
        angle2 = (np.degrees(np.arctan2(line2[1], line2[0])) + 360) % 360
    return np.random.uniform(min(angle1, angle2), max(angle1, angle2))
    # return (uniform(min(angle1, angle2), max(angle1, angle2)) + 2 * np.pi) % (2*np.pi)


def polysides(polygon):
    """
    Returns the sides of the polygon
    described as line segments.

    e.g

    For the polygon [p1, p2, p3], the result will be
    [[p1, p2], [p2, p3], [p3, p1]
    """
    return np.c_[polygon, np.roll(polygon, -1, axis=0)]


# Checks if a polygon is self-intersecting, by checking if one pair of its non-neighboring edges intersects
def check_self_intersection(poly):
    edges = polysides(poly)
    for i, edge1 in enumerate(edges):
        for edge2 in edges[i + 2:-1]:
            if check_edges_intersection(edge1, edge2):
                return True
        if (i != 0) and (i < len(edges) - 2):
            if check_edges_intersection(edge1, edges[-1]):
                return True
    return False


# Checks if 2 edges (of type [x1 y1 x2 y2]) intersect by comparing points triplets' orientation and covering general
# and special cases
def check_edges_intersection(edge1, edge2):
    p1 = [edge1[0], edge1[1]]
    p2 = [edge1[2], edge1[3]]
    p3 = [edge2[0], edge2[1]]
    p4 = [edge2[2], edge2[3]]

    orientation1_2_3 = check_orientation(p1, p2, p3)
    if (orientation1_2_3 == 0) and on_segment(p1, p2, p3):  # Checks special case
        return True
    orientation1_2_4 = check_orientation(p1, p2, p4)
    if (orientation1_2_4 == 0) and on_segment(p1, p2, p4):  # Checks special case
        return True
    orientation3_4_1 = check_orientation(p3, p4, p1)
    if (orientation3_4_1 == 0) and on_segment(p3, p4, p1):  # Checks special case
        return True
    orientation3_4_2 = check_orientation(p3, p4, p2)
    if (orientation3_4_2 == 0) and on_segment(p3, p4, p2):  # Checks special case
        return True

    # General cases
    if (orientation1_2_3 != orientation1_2_4) and (orientation3_4_1 != orientation3_4_2):
        return True

    return False


# Checks the orientation between 3 given points (by comparing each pair of points' slope). Returns 1 for clockwise,
# 0 for collinear, -1 for counter-clockwise
def check_orientation(p1, p2, p3):
    val = (float(p2[1] - p1[1]) * float(p3[0] - p2[0])) - (float(p2[0] - p1[0]) * float(p3[1] - p2[1]))
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0


# Given 3 collinear points, checks if p3 is on the line segment [p1,p2]
def on_segment(p1, p2, p3):
    if ((p3[0] <= max(p2[0], p1[0])) and (p3[0] >= min(p2[0], p1[0])) and (p3[1] <= max(p2[1], p1[1])) and (
            p3[1] >= min(p2[1], p1[1]))):
        return True
    return False


def erode_image(image, noise_volume=0, erosion_type="reference", noise_reference="average_edge_length"):
    """
    Erodes an image by applying noise on it.
    :param image: The image to be eroded.
    :param noise_volume: The volume of noise to be applied.
    :param erosion_type: The type of erosion to be applied. Can be 'reference' or 'sandstorm'.
    :param noise_reference: The reference to be used for calculating the noise volume, in case of 'reference' erosion type. Can be 'diameter', 'average_edge_length' or 'min_edge_length'.
    """

    if erosion_type == "reference":
        contours = find_contours(image[:, :, 3], 0)
        contours = [contour[:, ::-1] for contour in contours]
        if len(contours) == 0:
            contours = [np.array([[0, 0], [0, image.shape[0]], [image.shape[1], image.shape[0]], [image.shape[1], 0]])]
        # if reference == "diameter":
        #     noise_reference = calculate_diameter(contours)
        # elif reference == "average_edge":
        #     noise_reference = calculate_average_edge_length(contours)
        # elif reference == "min_edge":
        #     noise_reference = calculate_min_edge_length(contours)
        noise = calculate_epsilon(contours, noise_volume, noise_reference)
        eroded_contours, erased = apply_noise_on_piece(contours[0], noise)
        if not erased:
            eroded_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(eroded_mask, pts=[eroded_contours.astype(np.int32)], color=(255))
            eroded_image = np.zeros(image.shape, dtype=np.uint8)
            eroded_image[:, :, 3] = eroded_mask
            eroded_image[:, :, :3] = image[:, :, :3]
            eroded_image[eroded_image[:, :, 3] == 0] = 0
            return eroded_image

    elif erosion_type == "sandstorm":
        return "Not implemented yet"
    else:
        raise ValueError('Invalid erosion type. Must be one of "reference" or "sandstorm".')