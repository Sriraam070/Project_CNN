%matplotlib inline
import os
import pandas as pd
import numpy as np
import cv2
import random
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from skimage import exposure
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import tqdm.notebook as tqdm
import random
import glob
import functools
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,ReLU
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Flatten, Dense, Activation, Reshape

# Load weights
def load_weights(model, yolo_weight_file):
    data = np.fromfile(yolo_weight_file, np.float32)
    data = data[4:]
    index = 0
    for layer in model.layers:
        shape = [w.shape for w in layer.get_weights()]
        if shape:
            kshape, bshape = shape
            bia = data[index:index + np.prod(bshape)].reshape(bshape)
            index += np.prod(bshape)
            ker = data[index:index + np.prod(kshape)].reshape(kshape)
            index += np.prod(kshape)
            layer.set_weights([ker, bia])

# Box class
class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()

# Box utilities
def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2.
    l2 = x2 - w2 / 2.
    left = max(l1, l2)
    r1 = x1 + w1 / 2.
    r2 = x2 + w2 / 2.
    right = min(r1, r2)
    return right - left

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0: return 0
    area = w * h
    return area

def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

def yolo_net_out_to_car_boxes(net_out, threshold=0.2, sqrt=1.8, C=20, B=2, S=7):
    class_num = 6
    boxes = []
    SS = S * S 
    prob_size = SS * C 
    conf_size = SS * B 
    
    probs = net_out[0: prob_size].reshape([SS, C])
    confs = net_out[prob_size: (prob_size + conf_size)].reshape([SS, B])
    cords = net_out[(prob_size + conf_size):].reshape([SS, B, 4])
    
    for grid in range(SS):
        for b in range(B):
            bx = Box()
            bx.c = confs[grid, b]
            bx.x = (cords[grid, b, 0] + grid % S) / S
            bx.y = (cords[grid, b, 1] + grid // S) / S
            bx.w = cords[grid, b, 2] ** sqrt 
            bx.h = cords[grid, b, 3] ** sqrt
            p = probs[grid, :] * bx.c
            
            if p[class_num] >= threshold:
                bx.prob = p[class_num]
                boxes.append(bx)
                
    boxes.sort(key=lambda b: b.prob, reverse=True)
    for i in range(len(boxes)):
        boxi = boxes[i]
        if boxi.prob == 0: continue
        for j in range(i + 1, len(boxes)):
            boxj = boxes[j]
            if box_iou(boxi, boxj) >= 0.4:
                boxes[j].prob = 0.
    boxes = [b for b in boxes if b.prob > 0.]
    
    return boxes

def draw_box(boxes, im, crop_dim):
    imgcv = im
    [xmin, xmax] = crop_dim[0]
    [ymin, ymax] = crop_dim[1]
    for b in boxes:
        h, w, _ = imgcv.shape
        left  = int((b.x - b.w/2.) * w)
        right = int((b.x + b.w/2.) * w)
        top   = int((b.y - b.h/2.) * h)
        bot   = int((b.y + b.h/2.) * h)
        left = int(left * (xmax - xmin) / w + xmin)
        right = int(right * (xmax - xmin) / w + xmin)
        top = int(top * (ymax - ymin) / h + ymin)
        bot = int(bot * (ymax - ymin) / h + ymin)

        left = max(left, 0)
        right = min(right, w - 1)
        top = max(top, 0)
        bot = min(bot, h - 1)
        thick = int((h + w) // 150)
        cv2.rectangle(imgcv, (left, top), (right, bot), (255, 0, 0), thick)

    return imgcv

# Image paths
path = "test_images/test6.jpg"

image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(15, 8))
plt.imshow(image)
plt.axis("off")

# Load vehicle and non-vehicle images
random.seed(0)
vehicle = sorted(glob.glob('Dataset/vehicles/*/*.png'))
nonvehicle = sorted(glob.glob('Dataset/non-vehicles/*/*.png'))
random.shuffle(vehicle)
random.shuffle(nonvehicle)

# Display sample images
_, (axs1, axs2) = plt.subplots(2, 10, figsize=(20, 4))
for ax in axs1:
    img = cv2.imread(random.choice(vehicle))[:, :, ::-1]
    ax.imshow(img)
for ax in axs2:
    img = cv2.imread(random.choice(nonvehicle))[:, :, ::-1]
    ax.imshow(img)

plt.show()
plt.close()

# Feature extraction functions
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9, pix_per_cell=8, 
                     cell_per_block=2, hog_channel=0, spatial_feat=True, 
                     hist_feat=True, hog_feat=True):
    features = []
    for file in imgs:
        file_features = []
        image = mpimg.imread(file)
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: 
            feature_image = np.copy(image)

        if spatial_feat:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel], 
                                                         orient, pix_per_cell, cell_per_block, 
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient, 
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    return features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualize=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

# Image comparison
def show_side_by_side(img1, img2, title1="Car", title2="Not Car"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    ax1, ax2 = axes
    ax1.imshow(img1)
    ax1.set_title(title1)
    ax2.imshow(img2)
    ax2.set_title(title2)
    plt.show()

cars = [cv2.imread(x)[:, :, ::-1] for x in vehicle]
notcars = [cv2.imread(x)[:, :, ::-1] for x in nonvehicle]

show_side_by_side(cars[0], notcars[0])

# Feature extraction and visualization
car = cv2.cvtColor(cars[0], cv2.COLOR_RGB2YUV)
notcar = cv2.cvtColor(notcars[0], cv2.COLOR_RGB2YUV)
car_features, car_hog_img = get_hog_features(car[:, :, 0], 9, 8, 2, vis=True, feature_vec=False)
notcar_features, notcar_hog_img = get_hog_features(notcar[:, :, 0], 9, 8, 2, vis=True, feature_vec=False)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
ax1.imshow(car)
ax1.set_title("Car (YUV)")
ax2.imshow(notcar)
ax2.set_title("Not Car (YUV)")
ax3.imshow(car_hog_img)
ax3.set_title("Car HOG")
ax4.imshow(notcar_hog_img)
ax4.set_title("Not Car HOG")
plt.show()

# Extracting features
color_space = 'YUV'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

car_features = extract_features(vehicle, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(nonvehicle, color_space=color_space, 
                                   spatial_size=spatial_size, hist_bins=hist_bins, 
                                   orient=orient, pix_per_cell=pix_per_cell, 
                                   cell_per_block=cell_per_block, 
                                   hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                   hist_feat=hist_feat, hog_feat=hog_feat)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)

# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split data into training and testing sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

# Print training info
print('Using spatial binning of:', spatial_size)
print('Using', hist_bins, 'histogram bins')
print('Using HOG orientation:', orient)
print('Using HOG pixels per cell:', pix_per_cell)
print('Using HOG cells per block:', cell_per_block)
print('Feature vector length:', len(X_train[0]))

# Train a Linear SVM classifier
svc = LinearSVC()
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Sliding window
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    boxes = []
    img = img.astype(np.float32) / 255
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                boxes.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    return boxes

def apply_sliding_window(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(image)
    img = image.astype(np.float32) / 255
    box_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    for box in box_list:
        cv2.rectangle(draw_img, box[0], box[1], (0, 0, 255), 6)
    return draw_img, box_list

# Test the sliding window approach
ystart = 400
ystop = 656
scale = 1.5

output_image, boxes = apply_sliding_window(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
plt.imshow(output_image)
plt.show()

# Process a video
def process_frame(frame):
    ystart = 400
    ystop = 656
    scale = 1.5
    out_frame, _ = apply_sliding_window(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    return out_frame

input_video = 'project_video.mp4'
output_video = 'output_video.mp4'

clip = VideoFileClip(input_video)
output_clip = clip.fl_image(process_frame)
output_clip.write_videofile(output_video, audio=False)
