# Coursework

## Running

### Requirements
Install requirements specified in `requirements.txt` (These are very standard).
E.g.
```sh
pip install -r requirements.txt
```


### Adding Data
Data folders can be specified as arguments, however example commands in this readme assume all data folders were put into a local `./data` folder, with the same structure as the example datasets. Specifically, for certain result tables to be correctly printed, annotation (if any) filenames *must* have the structure of `test_image_<id>.txt`, and all images (e.g., query templates, test images) *must* have a `.png` extension.

---

## Task 1
Can be run from `task1.py`. Evaluates any given test samples from a given data directory which *must* be specified via `-d` or `--dir` and contain `list.txt`. E.g.

```sh
python task1.py -d "data/angle"
```

Test results are logged on finish like the following:
```
[O] [image1.png] - Truth: 40 | Unrounded Est: 40.19 | Rounded Est: 40
[O] [image2.png] - Truth: 20 | Unrounded Est: 20.09 | Rounded Est: 20
[O] [image3.png] - Truth: 60 | Unrounded Est: 60.05 | Rounded Est: 60
[O] [image4.png] - Truth: 120 | Unrounded Est: 120.15 | Rounded Est: 120
[O] [image5.png] - Truth: 53 | Unrounded Est: 52.72 | Rounded Est: 53
[O] [image6.png] - Truth: 72 | Unrounded Est: 72.01 | Rounded Est: 72
[O] [image7.png] - Truth: 95 | Unrounded Est: 94.96 | Rounded Est: 95
[O] [image8.png] - Truth: 109 | Unrounded Est: 108.89 | Rounded Est: 109
[O] [image9.png] - Truth: 18 | Unrounded Est: 18.02 | Rounded Est: 18
[O] [image10.png] - Truth: 90 | Unrounded Est: 90.00 | Rounded Est: 90

Tests completed | Passed: 10 | Failed: 0
```

Additionally, visualisations of the edgemapping and line-transform process can be exported to a given folder by specifying `-vdir` or `--visualise-dir`. E.g.

```sh
python task1.py -d "data/angle" -vdir "export/task1"
```

---

## Task 2
Can be run from `task2.py`. Must have the following arguments specified:


|Required Arg|Description|
|:--|:--|
|`--train-dir`|Training data directory containing **.png** image templates.|
|`--test-dir`|Test data directory containing **.png** test images.|
|`--anno-dir`|Test annotation directory containing **.txt** files with comma separated values: `{template name}, {min_bound}, {max_bound}`.|

E.g.
```sh
python task2.py --test-dir "data/Task2Dataset/TestWithoutRotations/images" --anno-dir "data/Task2Dataset/TestWithoutRotations/annotations" --train-dir "data/Task2Dataset/Training/png"
```

---

## Task 3
Task 3 involves identifying icons in each test image. This approach uses SIFT to compute keypoints and descriptors, and matches them using BF or FLANN based matcher.

Can be run from `task3_new.py`. There are no mandatory arguments. Below are a list of optional arguments:


|Optional Arg|Description|
|:--|:--|
|`--test_dir`, `-t`|Test data directory containing **.png** test images.|
|`--query-dir`, `-q`|Training data directory containing **.png** image templates.|
|`--anno_dir`, `-a`|Test annotation directory containing **.txt** files with comma separated values: `{template name}, {min_bound}, {max_bound}`.|
|`--res_dir`, `-r`|Results directory containing **.txt** predicted annotation files.|
|`--out_dir`, `-o`|Directory for outputting result images with bounding boxes and labels.|
|`--matcher`, `-m`|The matcher to use when matching descriptors. E.g. bf, flann|
|`--verbose`, `-v`|Print extra information when matching.|

E.g.
```sh
python task3.py -v
```

Test results (example) are logged on finish like the following:

```
[test_image_1 ] - Actual: 4 | Correct: 4 | Missed: 0 | Extra: 0
[test_image_10] - Actual: 4 | Correct: 4 | Missed: 0 | Extra: 0
[test_image_11] - Actual: 4 | Correct: 4 | Missed: 0 | Extra: 0
[test_image_12] - Actual: 4 | Correct: 4 | Missed: 0 | Extra: 0
[test_image_13] - Actual: 4 | Correct: 4 | Missed: 0 | Extra: 0
[test_image_14] - Actual: 5 | Correct: 4 | Missed: 1 | Extra: 0
[test_image_15] - Actual: 4 | Correct: 4 | Missed: 0 | Extra: 0
[test_image_16] - Actual: 4 | Correct: 4 | Missed: 0 | Extra: 0
[test_image_17] - Actual: 4 | Correct: 4 | Missed: 0 | Extra: 0
[test_image_18] - Actual: 5 | Correct: 5 | Missed: 0 | Extra: 0
[test_image_19] - Actual: 4 | Correct: 4 | Missed: 0 | Extra: 0
[test_image_2 ] - Actual: 4 | Correct: 4 | Missed: 0 | Extra: 0
[test_image_20] - Actual: 4 | Correct: 4 | Missed: 0 | Extra: 0
[test_image_3 ] - Actual: 4 | Correct: 4 | Missed: 0 | Extra: 0
[test_image_4 ] - Actual: 5 | Correct: 5 | Missed: 0 | Extra: 0
[test_image_5 ] - Actual: 5 | Correct: 4 | Missed: 1 | Extra: 0
[test_image_6 ] - Actual: 5 | Correct: 5 | Missed: 0 | Extra: 0
[test_image_7 ] - Actual: 5 | Correct: 5 | Missed: 0 | Extra: 0
[test_image_8 ] - Actual: 4 | Correct: 4 | Missed: 0 | Extra: 0
[test_image_9 ] - Actual: 4 | Correct: 4 | Missed: 0 | Extra: 0

Matching completed | Total icons: 86 | Correct guesses: 84 | Missed: 2 | False +ives: 0
Precision: 0.977
```
