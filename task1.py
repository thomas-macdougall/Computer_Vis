import cv2
import argparse
import numpy as np
import pandas as pd
import os

# TODO:
# 1. Hough Transform Visualisation & Justification of params
# 2. It would be pretty easy to make a test-case generator for this...

def dist(x, y, pt):
    return np.sqrt((x - pt[0])**2 + (y - pt[1])**2)

def get_line_gradients(lines, intersection=None):
    grads = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if intersection is None:
            r_pts = [[x1, y1], [x2, y2]]
        else:
            # Get correct line gradient direction by sorting points by distance from intersection.
            # Idx 0 is closest to intersection, 1 is farthest.
            r_pts = None
            if dist(x1, y1, intersection) > dist(x2, y2, intersection):
                r_pts = [[x2, y2], [x1, y1]]
            else:
                r_pts = [[x1, y1], [x2, y2]]

        # x, y = x2 - x1, y1 - y2 # y1 - y2 as convention in cv2 is for y to increase when going down visually.
        x, y = r_pts[1][0] - r_pts[0][0], r_pts[0][1] - r_pts[1][1] # y1 - y2 as convention in cv2 is for y to increase when going down visually.
        theta = np.arctan2(y, x) # Get polar theta in range 0 < t < 2pi
        grads.append(theta)

    return np.array(grads, dtype=np.float32)

def get_line_intersection(p1, p2, p3, p4):

    # Line 1
    a11 = (p1[1] - p2[1])
    a12 = (p2[0] - p1[0])
    b1 = (p1[0]*p2[1] - p2[0]*p1[1])

    # Line 2
    a21 = (p3[1] - p4[1])
    a22 = (p4[0] - p3[0])
    b2 = (p3[0]*p4[1] - p4[0]*p3[1])

    # Setup and solve system
    A = np.array([[a11, a12],
                [a21, a22]])

    b = -np.array([b1,
                b2])

    try:
        intersection_point = np.linalg.solve(A,b)
        return intersection_point

    except np.linalg.LinAlgError:
        # Lines are parallel
        return None

def draw_lines(img, lines, thickness=1, colour=(0,0,255)):
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), colour, thickness)

    return img

def draw_intersections(img, intersections):
    for i in intersections:
        x, y = i
        pos = (int(np.rint(x)), int(np.rint(y)))
        cv2.drawMarker(img, pos, (0,255,0), cv2.MARKER_CROSS, 5, 1)
        
    cv2.drawMarker(img, pos, (255,0,0), cv2.MARKER_CROSS, 15, 1)

    return img

def get_grads(lines, intersection=None):
    grads = get_line_gradients(lines, intersection)
    grads = np.expand_dims(grads, axis=1) # For use in k-means

    # K-means cluster line gradients. Expecting 2 lines so k=2. This helps get a better estimate in edge cases where line detection throws back more than 4 lines total.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) # The k-means doesn't have to be too precise as generally the line gradients should be distinct.
    _, labels, centers = cv2.kmeans(np.expand_dims(grads, axis=1), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return labels, centers

def get_angle(filename, save_dir=None):

    export = True if not save_dir is None else False

    raw = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    if export:
        cv2.imwrite(os.path.join(save_dir, "vis_raw.png"), raw)

    # Grayscale the image. Standard Canny procedure.
    gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    if export:
        cv2.imwrite(os.path.join(save_dir, "vis_gray.png"), gray)

    # Gaussian blur of moderate strength. Standard Canny proceure though is not explicitly needed for the given examples.
    #   Enough to filter out high frequency noise if needed, but not enough to lose the line edges.
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    if export:
        cv2.imwrite(os.path.join(save_dir, "vis_blur.png"), blur)

    # Use Otsu's method to pick out canny thresholds
    hi, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    lo = 0.5 * hi

    # Canny. Threshold determined via experimentation, aiming to produce a crisp outline on the example images.
    edges = cv2.Canny(blur, lo, hi)
    if export:
        cv2.imwrite(os.path.join(save_dir, "vis_edges.png"), edges)

    # First Iteration of line gradient calculation; find lines intersection point (or extrapolated intersection point) to determine where the acute angle should be. 
    #   This influences the secondary gradient calculation.
    # We cannot simply calculate acute angle of intersection as there is ambiguity based on where the line propagates to from the intersection (2 options).

    # Use HoughLinesP to get line *segments*. This resolves ambiguity between lines in opposite quadrants about the intersection point.
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=10, maxLineGap=250)

    # Make inital prediction about the line gradients (without incorporating if a line is radiating out from the intersection centre or not, so it may need to be 'flipped')
    labels, centers = get_grads(lines)

    lines_1 = lines[labels == 0]
    lines_2 = lines[labels == 1]

    if export:
        out = draw_lines(raw, lines_1)
        out = draw_lines(out, lines_2, colour=(255,255,0))
        cv2.imwrite(os.path.join(save_dir, "vis_lines.png"), out)

    # Get intersection points (or extrapolated intersection points) between different 'sublines' (edges) of the two major lines.
    intersections = []

    for l1 in lines_1:
        for l2 in lines_2:
            intersection = get_line_intersection([l1[0], l1[1]], [l1[2], l1[3]], [l2[0], l2[1]], [l2[2], l2[3]])
            if not intersection is None:
                intersections.append(intersection)

    if len(intersections) == 0: 
        return 0 # Lines must be parallel

    avg_intersection = np.mean(intersections, axis=1)

    # Refine gradient prediction based on calculated intersection point (fix ambiguous cases).
    labels, centers = get_grads(lines, avg_intersection)
    difference = min(np.abs(centers[0] - centers[1]), np.abs(centers[1] - centers[0])) # Get acute angle in rads
    difference = np.degrees(difference)

    if export:
        cv2.imwrite(os.path.join(save_dir, "vis_lines.png"), draw_intersections(raw, intersections))

    return difference

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Task 1")
    parser.add_argument('-d', '--dir', type=str, required=True, help="Data directory (Absolute or relative to running directory). Must contain file 'list.txt' which contains comma separtated {filename, angle} pairs. Filenames must be within the data directory.")
    parser.add_argument('-vdir', '--visualise-dir', type=str, default=None, help="Export directory for visualisation images (enables visualisation if given).")

    args = parser.parse_args()

    data = np.array(pd.read_csv(os.path.join(args.dir, "list.txt"), header=None))

    filenames, angles = data[:, 0], data[:, 1]

    # Setup visualise output dir if requested.
    VISUALIZE_IDX = 0
    if args.visualise_dir is not None:
        if not os.path.exists(args.visualise_dir):
            os.makedirs(args.visualise_dir)
        
    print(f"Beginning tests on data in {args.dir}.", end="\n\n")

    estimates = []
    failures = 0
    within_one_degree = 0
    for i, (f, a) in enumerate(zip(filenames, angles)):

        f = os.path.join(args.dir, f)

        if i == VISUALIZE_IDX:
            angle = get_angle(f, args.visualise_dir)[0]
        else:
            angle = get_angle(f)[0]        

        rounded_angle = int(np.rint(angle))
        estimates.append(rounded_angle)

        test_passed = rounded_angle == a
        if not test_passed:
            failures += 1
            if np.abs(rounded_angle - a) < 2:
                within_one_degree += 1

        print(f"[{'O' if rounded_angle == a else 'X'}] [{f}] - Truth: {a} | Unrounded Est: {angle:.2f} | Rounded Est: {rounded_angle}")

    test_result_out = f"Tests completed | Passed: {len(filenames) - failures} | Failed: {failures}"
    one_degree_out = f" | Of which <1deg out: {within_one_degree} "
    print("\n" + test_result_out + ("" if within_one_degree == 0 else one_degree_out))
