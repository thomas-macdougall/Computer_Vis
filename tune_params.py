from hyperopt import fmin, tpe, hp, STATUS_OK
from task3_new import preprocess, downscale, extract_labels, match_segmented, compute_keypoints_n_descriptors
from task3_segment import segment_images
import cv2
import os
from collections import Counter

def loss_function(res, truth):
    correct = 0
    res = Counter(res)
    truth = Counter(truth)
    for guess in truth:
        if guess in res:
            # don't penalise missed guesses
            if truth[guess] >= res[guess]:
                correct += res[guess]
            else:
                correct += truth[guess] 
                correct -= res[guess] - truth[guess] # penalise extra guesses
    return correct

def objective(params):
    # unpack parameters
    nfeatures = int(params['nfeatures'])
    nOctaveLayers = int(params['nOctaveLayers'])
    contrastThreshold = params['contrastThreshold']
    edgeThreshold = params['edgeThreshold']
    sigma = params['sigma']
    loweRatio = params['loweRatio']
    inliers = int(params['inliers'])
    ransacThresh = params['ransacThresh']

    # constants
    segmentedTests = params['tests']
    testLabels = params['testLabels']
    templates = params['templates']
    templateLabels = params['templateLabels']
    matcher = params['matcher']
    
    sift = cv2.SIFT_create(nfeatures=nfeatures, 
                           nOctaveLayers=nOctaveLayers,
                           contrastThreshold=contrastThreshold,
                           edgeThreshold=edgeThreshold,
                           sigma=sigma)

    loss = 0
    for i, segments in enumerate(segmentedTests):
        res = match_segmented(queries=templates, 
                              queryLabels=templateLabels,
                              segments=segments,
                              sift=sift, 
                              matcher=matcher,
                              ratio=loweRatio, 
                              ransacThreshold=ransacThresh, 
                              minInliers=inliers)
        res = [r[0] for r in res]
        truth = testLabels[i]
        loss += loss_function(res, truth)
    return -loss/len(testLabels) # minimise negative loss

    



templateLabels, templates = preprocess("./data/Task2Dataset/Training/png")
queries = downscale(templates)
_, tests = preprocess("./data/Task3Dataset/images")


annotations = os.listdir("./data/Task3Dataset/annotations")
testLabels = []
for i, f in enumerate(annotations):
    path = os.path.join("./data/Task3Dataset/annotations", f)
    labels = extract_labels(path)
    testLabels.append(labels)

segmentedTests = segment_images(tests, resize=True, save=False)

space = {
    'nfeatures': hp.quniform('nfeatures', 400, 2000, 10),
    'nOctaveLayers': hp.quniform('nOctaveLayers', 2, 8, 2),
    'contrastThreshold': hp.loguniform('contrastThreshold', -4, -1),
    'edgeThreshold': hp.uniform('edgeThreshold', 13, 19),
    'sigma': hp.uniform('sigma', 1, 5),
    'loweRatio': hp.uniform('loweRatio', 0, 1),
    'inliers': hp.quniform('inliers', 0, 10, 2),
    'ransacThresh': hp.uniform('ransacThresh', 0, 12), 
    'tests': segmentedTests,
    'testLabels': testLabels,
    'templates': queries,
    'templateLabels': templateLabels,
    'matcher': cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
}

best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    verbose=2
)

print(best)