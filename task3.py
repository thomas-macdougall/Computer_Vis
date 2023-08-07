import re
import numpy as np
import cv2

import argparse
import os
from collections import Counter

from task3_segment import segment_images

# Helper function to extract labels
def extract_labels(file):
    labels = []
    with open(file, "r") as f:
        lines = f.readlines()
    for line in lines:
        labels.append(line.split(',')[0])
    return labels

# Helper function to print results
def print_results(results_dir, anno_dir):
    result_files = [f for f in os.listdir(results_dir)]
    anno_files = [f for f in os.listdir(anno_dir)]
    if len(result_files) != len(anno_files):
        raise ValueError("Number of result files does not match annotation files")
    results = []
    total_missed, total_fp, total_tp = 0, 0, 0
    total_truth = 0
    for i in range(len(result_files)):
        predicted = Counter(extract_labels(os.path.join(results_dir, result_files[i])))
        truth = Counter(extract_labels(os.path.join(anno_dir, anno_files[i])))
        
        n_icons = sum(truth.values())
        total_truth += n_icons
        tp, fp, missed = 0, 0, 0

        if predicted == truth:
            results.append([result_files[i][:-4], n_icons, sum(predicted.values()), 0, 0])
            tp += n_icons
        else:
            for key in truth:
                if key not in predicted:
                    missed += truth[key]
                else:
                    if truth[key] > predicted[key]:
                        tp += predicted[key]
                        missed += truth[key] - predicted[key]
                    else:
                        fp += predicted[key] - truth[key]
                        tp += truth[key] - (predicted[key] - truth[key])
            results.append([result_files[i][:-4], n_icons, tp, fp, missed])
        total_missed += missed
        total_fp += fp
        total_tp += tp
    
    for r in results:
        indent = '' if len(r[0]) >= 13 else ' '
        print(f"[{r[0] + indent}] - Actual: {r[1]} | Correct: {r[2]} | Missed: {r[4]} | Extra: {r[3]}")
    
    print(f"\nMatching completed | Total icons: {total_truth} | Correct guesses: {total_tp} | Missed: {total_missed} | False +ives: {total_fp}")
    print(f"Precision: {total_tp / (total_truth):.3f}")

# Helper function to export results
def export_results(results_dir, testLabel, results, ext='.txt'):
    testLabel = testLabel + ext
    out_dir = os.path.join(results_dir, testLabel)
    with open(out_dir, "w") as f:
        for i in range(len(results)):
            f.write(', '.join(results[i]) + '\n')
        f.close()

# Helper function to draw matches between query and test image
def draw_matches(queryImage, queryKps, trainImage, trainKps, matches):
    image = cv2.drawMatchesKnn(queryImage, queryKps, trainImage, trainKps, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("matches", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Downscale an array of images
def downscale(images, scale=0.5):
    downscaled = []
    for image in images:
        image_d = cv2.resize(image.copy(), (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        downscaled.append(image_d)
    return downscaled

# Removes alpha channel from images
def remove_alpha(image):
    if image.shape[2] != 4:
        return image

    col = image[:,:,:3]
    alpha = image[:,:,3]

    background = np.ones_like(col, dtype=np.uint8) * 255

    alpha_blend = alpha[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_blend = np.concatenate((alpha_blend,alpha_blend,alpha_blend), axis=2)

    col = col.astype(np.float32) * alpha_blend
    background = background.astype(np.float32) * (1 - alpha_blend)

    return (col + background).astype(np.uint8)

# Preprocess images stored in given directory
def preprocess(files_dir):
    files = [f for f in os.listdir(files_dir) if f.endswith(".png")]
    processed = []
    labels = []
    for f in files:
        labels.append(os.path.splitext(f)[0])
        f = os.path.join(files_dir, f)
        image = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        image = remove_alpha(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to gray image
        processed.append(image)
    return labels, processed

# Compute keypoints and descriptors using opencv's SIFT
def compute_keypoints_n_descriptors(sift, images):
    keypoints_n_descriptors = []
    for image in images:
        keypoints, descriptors = sift.detectAndCompute(image, None)
        keypoints_n_descriptors.append((keypoints, descriptors))
    return keypoints_n_descriptors

# Match query images to each segment of a test image
def match_segmented(queries, queryLabels, segments, sift, matcher, ratio, ransacThreshold, minInliers, min_match_count=4):
    query_kds = compute_keypoints_n_descriptors(sift, queries)

    test_matches = []

    for segment, co_ords in zip(*segments):
        seg_kps, seg_des = compute_keypoints_n_descriptors(sift, [segment])[0]
        for query_id, (kps, des) in enumerate(query_kds):
            matches = matcher.knnMatch(des, seg_des, k=2)
            good_matches = []

            # Lowe's ratio test
            for m, n in matches:
                if m.distance < ratio * n.distance:
                    good_matches.append([m])
            
            if len(good_matches) > min_match_count:
                src_pts = np.float32([kps[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([seg_kps[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Compute homography using keypoints
                H, mask = cv2.findHomography(srcPoints=src_pts, dstPoints=dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransacThreshold)
                
                # Perform inlier check
                inlier_matches = 0
                for i in range(len(good_matches)):
                    if mask[i] == 1:
                        inlier_matches += 1
                    
                
                if inlier_matches < minInliers: # not a match
                    continue
                else:
                    if H is not None:
                        x1, y1, x2, y2 = co_ords

                        label = '-'.join(queryLabels[query_id].split('-')[1:])

                        test_matches.append([label, 
                                            str((x1, y1)), # top left corner
                                            str((x2, y2)), # bottom right corner
                                            inlier_matches]) 
    return test_matches

# draw bounding boxes on test images
def draw_bounding_boxes(img, file):
    # read each line of the file
    with open(file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.replace('(', '').replace(')', '').replace(' ', '').replace('\n', '')
        # split on multiple different delimiters
        line = line.split(',')
        label = line[0]
        top_left = (int(line[1]), int(line[2]))
        bottom_right = (int(line[3]), int(line[4]))

        # draw bounding boxes
        img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        img = cv2.putText(img, label, (top_left[0], bottom_right[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return img

# view the results
def view_results(results_dir, test_img_dir, output_dir):
    for file in os.listdir(results_dir):

        image_file = file.replace('.txt', '.png')

        # read the image
        img = cv2.imread(os.path.join(test_img_dir, image_file))
        
        # file location
        file = os.path.join(results_dir, file)

        # draw bounding boxes
        img = draw_bounding_boxes(img, file)

        # save the image
        cv2.imwrite(os.path.join(output_dir, image_file), img)

# check if there is a better match for a bounding box
def check_for_better_match(test_matches):
    final_matches = []
    
    # loop through each test image and find the best match for each bounding box
    for i in range(len(test_matches)):
        match = test_matches[i]
        to_add = True
        for j in range(len(test_matches)):
            comparison = test_matches[j]
            if (match[1] == comparison[1] and match[2] == comparison[2]):
                if match[3] < comparison[3]:
                    to_add = False
                    break
        if to_add:
            final_matches.append(match[:-1])

    return final_matches

# Main driver code
def main(args, **kwargs):
    # preprocess query templates and testing images
    queryLabels, queries = preprocess(args.query_dir)
    queries = downscale(queries)
    testLabels, test = preprocess(args.test_dir)

    # Segment test images
    # Each image contains several icons. Each icon is an image segment.
    # Output: List of lists, where each inner list contains icon segments, and each outer list denotes a test image
    segmentedTests = segment_images(test, resize=True, save=True)
    
    # Unpack hyperparameters
    sift = kwargs['sift']
    matcher = kwargs['matcher']
    ratio = kwargs['loweRatio']
    ransacThreshold = kwargs['RANSAC']
    minInliers = kwargs['inliers']

    # Template matching
    for i, segments in enumerate(segmentedTests):
        res = match_segmented(queries=queries, 
                        queryLabels=queryLabels,
                        segments=segments,
                        sift=sift, 
                        matcher=matcher,
                        ratio=ratio, 
                        ransacThreshold=ransacThreshold, 
                        minInliers=minInliers)
        
        res = check_for_better_match(res)

        export_results(args.res_dir, testLabels[i], res)
        if args.verbose:
            print(f"{testLabels[i]} done. Matched {len(res)} / {len(segments)} icons")
        else:
            print(f"Matching test {i+1}/{(20)}", end='\r', flush=True)
    
    print_results(args.res_dir, args.anno_dir)

    view_results(args.res_dir, args.test_dir, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", "-t", type=str, default="./data/Task3Dataset/images", help="Directory of training images")
    parser.add_argument("--query_dir", "-q", type=str, default="./data/Task2Dataset/Training/png", help="Directory of query images (base templates)")
    parser.add_argument("--anno_dir", "-a", type=str, default="./data/Task3Dataset/annotations", help="Directory of training annotations")
    parser.add_argument("--res_dir", "-r", type=str, default="./data/results", help="Directory for storing match results")
    parser.add_argument("--output_dir", "-o", type=str, default="./data/results_images", help="Directory for storing image results")
    parser.add_argument("--matcher", "-m", type=str, default="bf", help="Type of matcher used")
    parser.add_argument("--verbose", '-v', action="store_true", help="Whether program is verbose whilst running. Default is false")

    args = parser.parse_args()

    matchers = {
        'bf': cv2.BFMatcher(cv2.NORM_L2, crossCheck=False),
        'flann': cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
    }

    hyperparameters = {
        "matcher": matchers[args.matcher],
        "sift": cv2.SIFT_create(nfeatures=1700, nOctaveLayers=10, contrastThreshold=0.0039052330228148877, edgeThreshold=16.379139206562137, sigma=2.2201211013686857),
        "loweRatio": 0.6514343913409797,
        "inliers": 6,
        "RANSAC": 11.110294305510669
    }
	
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    main(args, **hyperparameters)

