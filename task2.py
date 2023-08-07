import argparse
import os
import numpy as np
import cv2
from time import perf_counter
import pandas as pd
import re

def normalise_image_to_range(img, min_val, max_val):
    # First, img to range -> 0, 1 then -> x, y
    img_min, img_max = np.min(img), np.max(img)
    img_diff = img_max - img_min
    normed = (img - img_min) / img_diff

    new_diff = max_val - min_val

    return ((normed * new_diff) + min_val).astype(np.float32)

# Util function to rotate an image *quickly* as alternatives (scipy.ndimage.rotate) are very slow
# Source: https://stackoverflow.com/a/47248339
def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def rotate_matches(matches):

    rotated = []
    for _, (cx, cy), (w, h, _), name in matches:
        rotation = int(name.rsplit("_", 1)[1][3:])
        a = rotation * np.pi / 180
        trans_mat = lambda x, y: np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])
        rot_matrix = np.array([
            [np.cos(a), np.sin(a), 0],
            [-np.sin(a), np.cos(a), 0],
            [0, 0, 1]
        ])

        dx, dy = w/2, h/2
        tl = np.array([cx - dx, cy - dy, 1])
        tr = np.array([cx + dx, cy - dy, 1])
        bl = np.array([cx - dx, cy + dy, 1])
        br = np.array([cx + dx, cy + dy, 1])

        transform = trans_mat(cx, cy) @ rot_matrix @ trans_mat(-cx, -cy)
        tl_r = transform @ tl
        tr_r = transform @ tr 
        bl_r = transform @ bl
        br_r = transform @ br

        rotated.append([tl_r[:2], tr_r[:2], br_r[:2], bl_r[:2]]) # This order to draw polygon correctly

    return np.array(rotated, dtype=np.int32)

def in_bounds(pos1, pos2, dims):
    # Unpack locs & dim
    x1, y1 = pos1
    x2, y2 = pos2
    w, h = dims[:2]

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Assuming pos1 is the middle of a bounding square
    if dx <= w / 2 and dy <= h / 2:
        return True

    return False

def template_match(img, templates, threshold=0.70, show_in_window=False, silent=False):

    matches = []

    # Template matching
    for temp_name, pyramid in templates.items():
        for temp in pyramid:

            correlation_vis = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)

            # if temp_name == "032-van_rot44":
            #     heatmap = cv2.applyColorMap((normalise_image_to_range(correlation_vis, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
            #     _, t = cv2.threshold(correlation_vis, 0.70, 1, cv2.THRESH_TOZERO)
            #     thresholded = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_JET)
                
            #     cv2.imwrite('heatmap.png', heatmap)
            #     cv2.imwrite('heatmap_threshold.png', thresholded)
            #     cv2.imshow('heatmap', heatmap)
            #     cv2.waitKey(0)

            THRESHOLD = threshold
            template_locations = np.where(correlation_vis >= THRESHOLD)

            ys, xs = template_locations
            for x, y in zip(xs, ys):
                center_x = x + temp.shape[0] // 2
                center_y = y + temp.shape[1] // 2
                matches.append((correlation_vis[y, x], (center_x, center_y), temp.shape, temp_name))

    matches.sort(key=lambda tup: tup[0], reverse=True)
    if not silent: 
        print(f"Found {len(matches)} Matches.")

    # Non-maximum suppression: Take max in image and store it as a template match; suppress all other correlation values in the bounds of that match. Repeat process until no values > threshold left.
    supressed = []

    # TODO: The runtime of this is probably terrible for large N, consider using a heap / priority queue if limiting.
    while len(matches) > 0:
        current_max = matches.pop(0) # Current max correlation.
        corr, loc, temp_size, temp_name = current_max

        supressed.append(current_max)

        for i in range (0, len(matches))[::-1]: # Traverse matches backwards so we can pop as we go without changing the next item idx.
            if in_bounds(loc, matches[i][1], temp_size):
                matches.pop(i)

    rotated_matches = rotate_matches(supressed)

    if show_in_window:
        # t_img_col = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        t_img_col = img
        for s, r in zip(supressed, rotated_matches):
            _, (x, y), shape, name = s
            # cv2.rectangle(t_img_col, (x - shape[0] // 2, y - shape[1] // 2), (x + shape[0] // 2, y + shape[1] // 2), (0,0,255), 1)
            cv2.drawMarker(t_img_col, (x,y), (0,255,0), cv2.MARKER_CROSS, 15, 1)
            cv2.putText(t_img_col, name, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1, 2)

            cv2.polylines(t_img_col, [r], 1, (255,255,0), 1)
            cv2.drawMarker(t_img_col, (int(r[0][0]),int(r[0][1])), (255,255,0), cv2.MARKER_CROSS, 15, 1)

        # cv2.imwrite("detected.png", t_img_col)
        cv2.imshow("detected", t_img_col)
        cv2.waitKey(0)
    
    return supressed

# This function can be used if you want to do gaussian blurring manually...
#   For now using cv2 is fine - otherwise use convolution theorem (fft), and this seperable kernel. E.g. ifft(fft(im) * fft(kx) * fft(ky))
def get_gaussian_kernel(r=5, sigma=1):

    coefficient = 1 / (np.sqrt(2 * np.pi) * sigma)
    # Generate linear space radius r centered about 0 - this is the samples from the gaussian distribution.
    kernel_space = np.linspace(-r, r, 5)
    exponents = np.exp(-np.square(kernel_space) / (2 * sigma ** 2))
    values = coefficient * exponents
    # Turn 1d kernel into 2d by constructing a matrix from k * k^T
    kernel_2d = np.outer(exponents, values)
    return kernel_2d / np.sum(kernel_2d)

def gaussian_blur(img, r=5, sigma=1, use_library_func=True):

    if use_library_func:
        return cv2.GaussianBlur(img, (r, r), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)

    k = get_gaussian_kernel(r, sigma)
    return cv2.filter2D(img, cv2.CV_64F, k, borderType=cv2.BORDER_DEFAULT)

def gaussian_pyramid_down(img, initial_size, min_size):
    
    dim = img.shape[:2]
    long_side = int(np.argmax(dim))
    side_ratio = dim[1 - long_side] / dim[long_side]

    if long_side == 0:
        initial_dim = (initial_size, int(initial_size * side_ratio))
    else:
        initial_dim = (int(initial_size * side_ratio), initial_size)

    last_octave = cv2.resize(img, initial_dim, interpolation=cv2.INTER_NEAREST)

    pyramid = [last_octave]

    while max(last_octave.shape[:2]) // 2 >= min_size:
        
        blur = gaussian_blur(last_octave, use_library_func=True)

        downscale = blur[::2, ::2]

        pyramid.append(downscale)
        last_octave = downscale

    return pyramid

def visualise_pyramid(pyramid, show_in_window=False):
    
    h1, w1 = pyramid[0].shape[:2]
    h2, w2 = pyramid[1].shape[:2]

    vis = np.zeros((h1, w1 + w2, 3), np.uint8)

    vis[:h1, :w1, :] = pyramid[0][:,:,:3]

    height_offset = 0
    for im in pyramid[1:]:
        h, w = im.shape[:2]
        vis[height_offset:height_offset+h, w1:w1+w, :] = im

        height_offset += h

    if show_in_window:
        cv2.imshow("pyramid", vis)
        cv2.waitKey(0)

    return pyramid

def rotate_templates(templates, n_rotations):
    
    if n_rotations == 0:
        return [(0, templates)]

    rotation_interval = 360 // n_rotations #deg

    all_rotations = [(0, templates)]

    # Don't do first (0) or last rotation as already accounted for by original template
    for i in range(1, n_rotations):
        angle = i * rotation_interval
        
        rotated = []

        for t in templates:
            rotated.append(rotate_image(t, angle))

        all_rotations.append((angle, rotated))

    return all_rotations

def apply_pyramid_schema(templates, initial_size, min_size, n_rotations):
    
    base_pyramids = []
    for name, t in templates:
        pyramid = gaussian_pyramid_down(t, initial_size, min_size)
        base_pyramids.append((name, pyramid))

    all_templates = {}
    for name, pyramid in base_pyramids:

        all_rotations = rotate_templates(pyramid, n_rotations) # Includes rot=0

        for rot, ts in all_rotations:
            all_templates[f"{name}_rot{rot}"] = ts
    return all_templates
    
def remove_alpha(im, background_col=[0,0,0]):
    if im.shape[2] != 4:
        return im

    col = im[:,:,:3]
    alpha = im[:,:,3]

    background = np.full_like(col, background_col, dtype=np.uint8)

    alpha_blend = alpha[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_blend = np.concatenate((alpha_blend,alpha_blend,alpha_blend), axis=2)

    col = col.astype(np.float32) * alpha_blend
    background = background.astype(np.float32) * (1 - alpha_blend)

    return (col + background).astype(np.uint8)

def euclidian_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1
    
    return np.sqrt(dx * dx + dy * dy)

def check_against_annotation(predictions, annotations, bound_threshold=10):

    # Note tn is simply the amount of unique templates not included in the annotation.
    tp, fp, fn = 0, 0, 0

    detected = []
    
    for pred in predictions:

        _, centre, dim, temp_name = pred

        base_template, rotation = temp_name.rsplit("_", 1)
        try:
            _, base_template = base_template.split("-", 1)
        except ValueError:
            raise NameError(f"ERROR: Expecting prefix on image template name (e.g. 000-Name), except got '{base_template}'.")
        rotation = rotation[3:]

        detected.append(base_template)

        if not base_template in annotations:
            fp += 1
            continue

        # TODO: This bound calculation needs to account for rotated images eventually.
        pred_p1 = (centre[0] - dim[0] // 2, centre[1] - dim[1] // 2)
        pred_p2 = (centre[0] + dim[0] // 2, centre[1] + dim[1] // 2)

        # Ensure predicted bounds are within a certain tolerance (Euclidian distance) of the actual bounds.
        d1 = euclidian_distance(pred_p1, annotations[base_template][0])
        d2 = euclidian_distance(pred_p2, annotations[base_template][1])

        if d1 <= bound_threshold and d2 <= bound_threshold:
            tp += 1
        else:
            fp += 1

    for temp in annotations.keys():
        if not temp in detected:
            fn += 1
    
    return tp, fp, fn

def get_dominant_colour(img):
    colors, count = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

def load_and_preprocess_templates(train_dir, col_to_match=None):

    files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f)) and f.endswith(".png")]

    templates = []
    
    for f in files:
        template_name = os.path.splitext(f)[0]

        f = os.path.join(train_dir, f)

        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if col_to_match is None:
            im = remove_alpha(im)
        else: 
            im = remove_alpha(im, col_to_match)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        templates.append((template_name, im))

    return templates

def load_and_preprocess_test_images(test_dir):

    files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f)) and f.endswith(".png")]

    test_imgs = []
    
    for f in files:
        test_name = os.path.splitext(f)[0]

        f = os.path.join(test_dir, f)

        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        col = remove_alpha(im)
        im = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)

        # blur = cv2.GaussianBlur(im, (7,7), 0)

        # # This threshold is not perfect and dilates the image a little...
        # _, thresh = cv2.threshold(blur, cv2.getTrackbarPos("t", "thresh"), 255, cv2.THRESH_BINARY_INV)

        # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        # eroded = cv2.erode(opening, np.ones((5,5), np.uint8), iterations=1)

        # masked = cv2.bitwise_and(col, col, mask=eroded)
    
        test_imgs.append((test_name, col))

    return test_imgs

def load_annotations(anno_dir):

    files = [f for f in os.listdir(anno_dir) if os.path.isfile(os.path.join(anno_dir, f)) and f.endswith(".txt")]

    annotations = {}

    for f in files:

        file_anno = {}

        img_name = os.path.splitext(f)[0]

        data = np.array(pd.read_csv(os.path.join(anno_dir, f), header=None))
        
        for ln in data:

            temp_name, x1, y1, x2, y2 = ln

            num_only = lambda n: int(re.sub("[^\d\.]", "", n))

            file_anno[temp_name] = [(num_only(x1), num_only(y1)), (num_only(x2), num_only(y2))]

        annotations[img_name] = file_anno

    return annotations

def main(train_dir, test_dir, anno_dir, rotations, max_temp_size, min_temp_size, threshold, silent):

    if max_temp_size < min_temp_size:
        raise ValueError(f"ERROR: Max template size ({max_temp_size}) is lower than minimum ({min_temp_size}).")

    if not silent: print("Generating template pyramids...")


    # Preprocess Test Images & Load annotations
    test_images = load_and_preprocess_test_images(test_dir)
    annotations = load_annotations(anno_dir)

    # Take example test image and use it to find the most dominant (background) colour.
    # Assuming all test images have the same bg, could regenerate pyramid each time to solve this issue.
    _, ex = test_images[0]
    bg_col = get_dominant_colour(ex)
    
    # Preprocess Templates
    base_templates = load_and_preprocess_templates(train_dir, bg_col)

    t0 = perf_counter()
    scaled_and_rotated_templates = apply_pyramid_schema(
        base_templates, 
        max_temp_size,
        min_temp_size,
        rotations
    )
    t1 = perf_counter()
    if not silent: print(f"Scaled and rotated {len(base_templates)} base templates ({len(scaled_and_rotated_templates)} pyramids generated) in {t1-t0:0.4f}s", end="\n\n")
    # visualise_pyramid(scaled_and_rotated_templates["016-house_rot110"], True)

    # Template Matching & Evaluation
    t_start = perf_counter()
    tot_tp, tot_fp, tot_fn = 0, 0, 0
    for i, (name, test) in enumerate(test_images):
        if not silent: print(f"[{i+1}/{len(test_images)}] Matching for image {name}...")
        t0 = perf_counter()
        matches = template_match(test, scaled_and_rotated_templates, threshold=threshold, show_in_window=False, silent=silent)
        t1 = perf_counter()

        # Check against annotation
        if not name in annotations:
            raise FileNotFoundError(f"ERROR: Could not find corresponding annotation file for [{name}] in the given annotation directory.")

        tp, fp, fn = check_against_annotation(matches, annotations[name])
        tot_tp += tp; tot_fp += fp; tot_fn += fn

        if not silent: print(f"Done ({t1-t0:0.4f}s). | Detected: {tp}/{len(annotations[name])} | Missed: {fn} | False Positives: {fp}", end="\n\n")

    t_end = perf_counter()
    acc = (tot_tp / (tot_tp + tot_fp)) if tot_tp + tot_fp != 0 else 0
    if not silent: 
        print(f"Finished. | Time elapsed: {t_end-t_start:0.4f}s | Average per image: {((t_end-t_start) / len(test_images)):0.4f}s.")
        print(f"Accuracy: {acc:0.4f} | Total Missed (FN): {tot_fn}")

    # (Total t, Average t, Acc, FN, tp, fp)
    return t_end-t_start, ((t_end-t_start) / len(test_images)), acc, tot_fn, tot_tp, tot_fp

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Task 2")
    parser.add_argument('--train-dir', type=str, required=True, help="Training data directory containing *png* image templates.")
    parser.add_argument('--test-dir', type=str, required=True, help="Test data directory containing *png* test images.")
    parser.add_argument('--anno-dir', type=str, required=True, help="Test annotation directory containing *txt* files with comma separated values: '{template name}, {min_bound}, {max_bound}'.")

    parser.add_argument('-r', '--rotations', type=int, default=10, help="Number of rotations applied to base templates during gaussian pyramid generation.")
    parser.add_argument('--max-template-size', type=int, default=128, help="Max template size. (Gaussian pyramid top level).")
    parser.add_argument('--min-template-size', type=int, default=32, help="Min template size. (Gaussian pyramid bottom level).")
    parser.add_argument('-t', '--threshold', type=float, default=0.70, help="Template matching detection threshold (0 < t < 1).")
    
    parser.add_argument('-s', '--silent', action='store_true', help="Supress print statements.")

    args = parser.parse_args()

    main(
        args.train_dir,
        args.test_dir,
        args.anno_dir,
        args.rotations,
        args.max_template_size,
        args.min_template_size,
        args.threshold,
        args.silent
    )