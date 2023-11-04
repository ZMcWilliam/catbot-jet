import math
import numpy as np
import cv2
from typing import List, Tuple

# Type aliases
Contour = List[List[Tuple[int, int]]]

g_last_line_pos = None # TODO: Integrate this properly as a param into the distToLastLine function

def distToLastLine(point: Tuple[int, int]) -> float:
    """
    Calculates the distance between a point and the last line position.

    Args:
        point (Tuple[int, int]): The point for which to calculate the distance.

    Global Args:
        g_last_line_pos (Tuple[int, int]): The last line position.

    Returns:
        float: The distance between the point and the last line position.
    """
    if point[0][0] > g_last_line_pos[0]:
        return np.linalg.norm(np.array(point[0]) - g_last_line_pos)
    return np.linalg.norm(g_last_line_pos - point[0])

# Vectorize the distance function so it can be applied to a numpy array
# This helps speed up calculations when calculating the distance of many points
distToLastLineFormula = np.vectorize(distToLastLine)

def findBestContours(contours: List[Contour], contour_thresh: int, last_line_pos: Tuple[int, int]) -> List[Contour]:
    """
    Processes a set of contours to find the best one to follow
    Filters out contours that are too small,
    then, sorts the remaining contours by distance from the last line position

    Args:
        contours (List[Contour]): The contours to be processed.
        contour_thresh (int): The minimum contour area to be considered valid.
        last_line_pos (Tuple[int, int]): The last known optimal line position.

    Returns:
        contour_values: A numpy array of contours, sorted by distance from the last line position
        [
            contour_area: float,
            contour_rect: cv2.minAreaRect,
            contour: np.array,
            distance_from_last_line: float
        ]
    """
    global g_last_line_pos
    # Create a new array with the contour area, contour, and distance from the last line position (to be calculated later)
    contour_values = np.array([[cv2.contourArea(contour), cv2.minAreaRect(contour), contour, 0] for contour in contours], dtype=object)

    # In case we have no contours, just return an empty array instead of processing any more
    if len(contour_values) == 0:
        return []

    # Filter out contours that are too small
    contour_values = contour_values[contour_values[:, 0] > contour_thresh]

    # No need to sort if there is only one contour
    if len(contour_values) <= 1:
        return contour_values

    # Sort contours by distance from the last known optimal line position
    g_last_line_pos = last_line_pos
    contour_values[:, 3] = distToLastLineFormula(contour_values[:, 1])
    contour_values = contour_values[np.argsort(contour_values[:, 3])]
    return contour_values

def centerOfContour(contour: Contour) -> Tuple[float, float]:
    """
    Calculates the center coordinates of a contour.

    Args:
        contour (np.array): The contour for which to calculate the center.

    Returns:
        Tuple[float, float]: The x and y coordinates of the contour's center.
    """
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

def centerOfLine(line: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[float, float]:
    """
    Calculates the center coordinates of a line segment.

    Args:
        line (Tuple[Tuple[int, int], Tuple[int, int]]): The line segment represented by two points.

    Returns:
        Tuple[float, float]: The x and y coordinates of the line segment's center.
    """
    return (int((line[0][0] + line[1][0]) / 2), int((line[0][1] + line[1][1]) / 2))

def pointDistance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Calculates the distance between two points.

    Args:
        p1 (Tuple[int, int]): The first point coordinates (x, y).
        p2 (Tuple[int, int]): The second point coordinates (x, y).

    Returns:
        float: The Euclidean distance between the two points.
    """
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def midpoint(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[float, float]:
    """
    Calculates the midpoint between two points.

    Args:
        p1 (Tuple[int, int]): The coordinates of the first point.
        p2 (Tuple[int, int]): The coordinates of the second point.

    Returns:
        Tuple[float, float]: The coordinates of the midpoint.
    """
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def simplifiedContourPoints(contour: Contour, epsilon: float = 0.01) -> List[Tuple[int, int]]:
    """
    Simplifies a given contour by reducing the number of points while maintaining the general shape

    Args:
        contour (Contour): The contour to be simplified.
        epsilon (float): The level of simplification. Higher values result in more simplification. Default is 0.01.

    Returns:
        List[Tuple[int, int]]: The simplified contour as a list of points.
    """
    epsilonBL = epsilon * cv2.arcLength(contour, True)
    return [pt[0] for pt in cv2.approxPolyDP(contour, epsilonBL, True)]

def getTouchingEdges(points: List[Tuple[int, int]], shape: Tuple[int, int]) -> List[str]:
    """
    Determines which edges of an image a given contour is touching.

    Args:
        points (list[Tuple[int, int]]): The list of points to check
        shape (Tuple[int, int]): The shape of the image. (height, width)

    Returns:
        List[str]: A list of strings (left, right, top, bottom) representing the edges that the contour is touching.
    """
    edges = []
    for point in points:
        if point[0] == 0 and "left" not in edges:
            edges.append("left")
        if point[0] == shape[1] - 1 and "right" not in edges:
            edges.append("right")
        if point[1] == 0 and "top" not in edges:
            edges.append("top")
        if point[1] == shape[0] - 1 and "bottom" not in edges:
            edges.append("bottom")
    return edges
