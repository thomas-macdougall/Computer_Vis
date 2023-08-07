import os
import cv2
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import argparse
import os
from time import strftime
import random
from PIL import Image

# TODO: The downscaling on icons here is pretty trash.

def load_icons(icon_dir):

    files = [f for f in os.listdir(icon_dir) if os.path.isfile(os.path.join(icon_dir, f)) and f.endswith(".png")]

    templates = []
    
    for f in files:
        file_name = os.path.splitext(f)[0]
        icon_name = file_name.split("-", 1)[1]

        f = os.path.join(icon_dir, f)

        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        templates.append((icon_name, im))

    return templates

def is_overlapping(bounding_boxes, box, t=0):
    # t = threshold
    # 0: x min, 1: x max, 2: y min, 3: y max
    for b in bounding_boxes:
        if b[1] + t < box[0] or b[0] > box[1] + t: # box on left side or right side
            continue
        
        if b[3] + t < box[2] or b[2] > box[3] + t: # box above or below
            continue
        return True

def can_fit(rectangles, w=512, h=512):

    w_r = w
    h_r = h
    for _, r in rectangles:
        r_w, r_h = r.shape[0], r.shape[1]
        if r_w <= w_r and r_h <= h_r:
            w_r -= r_w
            h_r -= r_h
        elif r_w <= w_r and r_h <= h:
            w_r -= r_w
        elif r_w <= w and r_h <= h_r:
            h_r -= r_h
        else:
            return False
    return True
        
def resize_long_side(img, size):
    dim = img.shape[:2]
    long_side = int(np.argmax(dim))
    side_ratio = dim[1 - long_side] / dim[long_side]

    if long_side == 0:
        img_dim = (size, int(size * side_ratio))
    else:
        img_dim = (int(size * side_ratio), size)

    return cv2.resize(img, img_dim, interpolation=cv2.INTER_NEAREST)

def sample_some_icons(icons, icon_sizes, icon_rotations, imin, imax):
        n_samples = np.random.randint(imin, imax + 1)
        # TODO: This can be choices (i.e. replacement) but would involve changing how the annotations are read in to make work for t2.
        sampled_icons = random.sample(icons, k=n_samples)
        sizes = random.choices(icon_sizes, k=n_samples)
        rotations = random.choices(icon_rotations, k=n_samples)

        # Resize (along long axis) and rotate
        icons = [(name, rotate(resize_long_side(ic, ic_s), ic_r)) for (name, ic), ic_s, ic_r in zip(sampled_icons, sizes, rotations)]
        return icons

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Task 2")
    parser.add_argument('--icon-dir', type=str, required=True, help="Icon directory containing *png* icons.")
    parser.add_argument('--out-dir', type=str, required=True, help="Test data directory containing *png* test images.")
    parser.add_argument('-width', type=int, default=512)
    parser.add_argument('-height', type=int, default=512)
    parser.add_argument('-r', '--rotations', type=int, default=360, help="Number of valid rotation intervals.")
    parser.add_argument('-smin', '--scale-min', type=int, default=64, help="Scale minimum.")
    parser.add_argument('-smax', '--scale-max', type=int, default=64, help="Scale maximum.")
    parser.add_argument('-smult', '--scale-mult', type=int, default=-1, help="Scale multiplier to space valid scales.")
    parser.add_argument('-imin', type=int, default=3)
    parser.add_argument('-imax', type=int, default=7)
    parser.add_argument('-n', type=int, default=30)
    parser.add_argument('--random-bg', action='store_true', help="Generate a random background colour for each image.")

    args = parser.parse_args()

    timestamp = strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(args.icon_dir):
        raise ValueError("ERROR: Invalid or non-existent icon directory given.")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    icons = load_icons(args.icon_dir)
    
    # Precompute valid icon sizes.
    if args.scale_mult < 0:
        icon_sizes = list(range(args.scale_min, args.scale_max+1))
    else:
        icon_sizes = []
        s = args.scale_max
        while s >= args.scale_min:
            icon_sizes.append(s)
            s //= args.scale_mult
    
    # Similarly with rotations
    icon_rotations = list(range(0, 360, 360 // args.rotations))

    for i in range(args.n):
        # Background Colour
        if args.random_bg:
            background_col = np.random.choice(range(256), size=3)
        else:
            background_col = np.zeros(3)
        background = np.full((args.height, args.width, 3), background_col, dtype=np.uint8)
        background = Image.fromarray(background)

        chosen_icons = sample_some_icons(icons, icon_sizes, icon_rotations, args.imin, args.imax)
        MAX_TRIES = 10000
        t = 0
        while not can_fit(chosen_icons, args.width, args.height) and t < MAX_TRIES:
            chosen_icons = sample_some_icons(icons, icon_sizes, icon_rotations, args.imin, args.imax)

        chosen_icons.sort(key=lambda x: x[1].shape[0] * x[1].shape[1], reverse=True)
        
        if not can_fit(chosen_icons, args.width, args.height):
            raise RuntimeError("ERROR: Could not fit icons into image in a reasonable number of tries.")

        bounding_boxes = [] #[(x_min, x_max, y_min, y_max)]
        for name, icon in chosen_icons:
            w, h = icon.shape[:2]
            x, y = np.random.randint(1, args.width - 2 - w), np.random.randint(1, args.height - 2 - h) # don't want it to touch exactly the boundary
            icon_dims = (x, x + w, y, y + h)

            if len(bounding_boxes) > 0:
                MAX_TRIES = 10000
                t = 0
                while is_overlapping(bounding_boxes, icon_dims, 5) and t < MAX_TRIES:
                    x, y = np.random.randint(1, args.width - 1 - w), np.random.randint(1, args.height - 1 - h)
                    icon_dims = (x, x + w, y, y + h)

                    if y + h > 512 or x + w > 512:
                        print(icon_dims)

                    t+=1

                if is_overlapping(bounding_boxes, icon_dims, 5):
                    raise RuntimeError("ERROR: Could not fit icons into image in a reasonable number of tries (at bbox step).")

            bounding_boxes.append(icon_dims)
            icon_img = Image.fromarray(icon)
            background.paste(icon_img, (x, y), icon_img)
        
        # Write files & Make output folders
        parent_folder = os.path.join(args.out_dir, timestamp)
        if not os.path.exists(parent_folder): os.makedirs(parent_folder)
        images_folder = os.path.join(parent_folder, "images")
        if not os.path.exists(images_folder): os.makedirs(images_folder)
        annotation_folder = os.path.join(parent_folder, "annotations")
        if not os.path.exists(annotation_folder): os.makedirs(annotation_folder)

        img_out_f = os.path.join(images_folder, f"test_image_{i + 1}.png")
        cv2.imwrite(img_out_f, np.array(background), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        anno_out_f = os.path.join(annotation_folder, f"test_image_{i + 1}.txt")
        with open(anno_out_f, 'w') as f:
            for i, (name, icon) in enumerate(chosen_icons):
                out = f"{name}, {bounding_boxes[i][0], bounding_boxes[i][2]}, {bounding_boxes[i][1], bounding_boxes[i][3]}\n" # name, top left, bottom right
                f.write(out)