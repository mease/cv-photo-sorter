###############################################
# Derek Mease
# CSCI 4831-5722 (Fleming)
# Final Project - Computer Vision Photo Sorter
###############################################

# This file contains code for sorting images based on similarity.

import numpy as np
import cv2 as cv
import pickle
import copyreg
from os import listdir, makedirs
from os.path import join, exists
from PyQt5.QtWidgets import QApplication
from sklearn.cluster import AffinityPropagation, DBSCAN
from sklearn.preprocessing import MinMaxScaler


# Helper for cacheing keypoint data.
def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                         point.response, point.octave, point.class_id)


copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)

# Group objects based on their histogram similarity.
#   https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
class HistogramGrouper:
    def __init__(self, img_path):
        self.img_path = img_path

    # Group similar photos in a list of photos.
    def group_similar(self, progress, files):
        hists = []

        h_bins = 100
        s_bins = 100
        histSize = [h_bins, s_bins]
        h_ranges = [0, 180]
        s_ranges = [0, 256]
        ranges = h_ranges + s_ranges
        channels = [0, 1]

        for f in files:
            progress.set_status(f'Computing histograms: {f}')
            img = cv.imread(join(self.img_path, f))
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            hist = cv.calcHist([hsv], channels, None, histSize, ranges, accumulate=False)
            cv.normalize(hist, hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            hists.append(hist)
            progress.increment(1)

        similarity = np.zeros((len(files), len(files)))
        for i in range(len(files)):
            progress.set_status(f'Comparing histograms: {files[i]}')
            for j in range(i, len(files)):
                similarity[i,j] = cv.compareHist(hists[i], hists[j], 0)
                similarity[j,i] = similarity[i,j]

        # Cluster with Affinity Propagation
        progress.set_status('Clustering')
        clustering = AffinityPropagation(affinity='precomputed', random_state=None).fit_predict(similarity)
        groups = []
        for c in set(clustering):
            groups.append([i for index, i in enumerate(files) if clustering[index] == c])
        return groups


# General purpose feature detector for grouping similar images.
class FeatureDetector:
    def __init__(self, img_path, data_path, detector, matcher, similarity_threshold=0.05):
        self.similarity_threshold = similarity_threshold
        self.data_path = data_path
        self.img_path = img_path
        self.sim_file = join(self.data_path, 'similarities.pickle')
        self.detector = detector
        self.matcher = matcher

        if not exists(self.data_path):
            makedirs(self.data_path)

    # Load cached features
    def feature_file_stored(self, img_name):
        files = [f for f in listdir(self.data_path)]
        return img_name + '.pickle' in files

    # Get features from image
    def get_features(self, img_name):
        files = [f for f in listdir(self.data_path)]

        if not self.feature_file_stored(img_name):
            img = cv.imread(join(self.img_path, img_name), cv.IMREAD_GRAYSCALE)

            # Resize large images for faster processing
            max_width = 2000
            height, width = img.shape
            if width > max_width:
                scale_percent = max_width / width
                width = int(width * scale_percent)
                height = int(height * scale_percent)
                dim = (width, height)
                img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
            else:
                scale_percent = 1

            # Detect features
            kp, des = self.detector.detectAndCompute(img, None)

            # Cache features
            with open(join(self.data_path, img_name + '.pickle'), 'wb') as f:
                pickle.dump([kp, des], f)

    # Detect and cache all features in a list of files
    def get_all_features(self, progress, files):
        # Calculate and store features
        for f in files:
            progress.set_status(f'Calculating features for {f}')
            self.get_features(f)
            progress.increment(1)

    # Compute similarity between two files based on features.
    def get_similarity(self, img1, img2):
        sim_dict = {}
        if exists(self.sim_file):
            with open(self.sim_file, 'rb') as f:
                sim_dict = pickle.load(f)

        if (img1, img2) in sim_dict.keys():
            return sim_dict[(img1, img2)]

        if not self.feature_file_stored(img1) or not self.feature_file_stored(img2):
            raise 'Feature file not stored for ' + img1

        with open(join(self.data_path, img1 + '.pickle'), 'rb') as f:
            kp1, des1 = pickle.load(f)
        with open(join(self.data_path, img2 + '.pickle'), 'rb') as f:
            kp2, des2 = pickle.load(f)

        if isinstance(self.matcher, cv.BFMatcher):
            matches = self.matcher.match(des1, des2)
        else:
            matches = self.matcher.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_points = []
        for match in matches:
            if len(match) == 2:
                if match[0].distance < 0.70 * match[1].distance:
                    good_points.append(match[0])

        # Calculate similarity (number of points matched divided by the union of points from both images)
        similarity = len(good_points) / (len(kp1) + len(kp2) - len(good_points))
        sim_dict[(img1, img2)] = similarity
        sim_dict[(img2, img1)] = similarity
        with open(self.sim_file, 'wb') as f:
            pickle.dump(sim_dict, f)
        return similarity

    # Group similar images in a list of images.
    def group_similar(self, progress, files, threshold=0.05):
        # Create similarity matrix
        similarity = np.zeros((len(files), len(files)))
        for i in range(len(files)):
            progress.set_status(f'Computing similarity: {files[i]}')
            for j in range(i, len(files)):
                if i == j:
                    similarity[i,j] = 1.0
                else:
                    similarity[i,j] = self.get_similarity(files[i], files[j])
                    similarity[j,i] = similarity[i,j]
            progress.increment(1)

        # Cluster with Affinity Propagation
        progress.set_status('Clustering')
        clustering = AffinityPropagation(affinity='precomputed',
                                         max_iter=1_000,
                                         convergence_iter=100,
                                         random_state=None).fit_predict(similarity)
        groups = []
        for c in set(clustering):
            groups.append([i for index, i in enumerate(files) if clustering[index] == c])
        
        # Affinity propagations gives a decent starting point for the groupings,
        # but it tends to mis-group several objects. Here we are checking all the
        # groups and removing any photos that have a similarity less than a particular
        # threshold compared to the other photos in the group.
        new_groups = []
        for group in groups:
            removed = []
            for i in range(len(group)):
                i_idx = files.index(group[i])
                sims = []
                for j in range(len(group)):
                    if (i != j):
                        j_idx = files.index(group[j])
                        sims.append(similarity[i_idx, j_idx])

                if max(sims) < threshold:
                    removed.append(files[i_idx])
            
            new_group = group
            for f in removed:
                new_groups.append([f])
                new_group.remove(f)
            new_groups.append(new_group)

        return new_groups


# SIFT feature detector
class Sift(FeatureDetector):
    def __init__(self, img_path, similarity_threshold=0.05):
        super().__init__(img_path,
                         './data/sift/',
                         cv.SIFT_create(nfeatures=500),
                         cv.FlannBasedMatcher(
                             dict(algorithm=1,  # FLANN_INDEX_KDTREE
                                  trees=5),
                             dict(checks=50)
                         ),
                         similarity_threshold)


# ORB feature detector
class Orb(FeatureDetector):
    def __init__(self, img_path, similarity_threshold=0.05):
        super().__init__(img_path,
                         './data/orb/',
                         cv.ORB_create(),
                         #cv.BFMatcher_create(cv.NORM_HAMMING, crossCheck=True),
                         cv.FlannBasedMatcher(
                             dict(algorithm=6,  # FLANN_INDEX_LSH
                                  table_number=12,
                                  key_size=20,
                                  multi_probe_level=2),
                             dict()
                         ),
                         similarity_threshold)
