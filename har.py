# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 19:07:40 2023

@author: Nurfitri Anbarsanti G2104045K
"""
# Bismillahirrahmanirrahim, EE6222 Assignment

import cv2
import numpy as np
import glob
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from array import array
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
from mlxtend.plotting import plot_decision_regions
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SimpleRNN
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from keras.utils import to_categorical

### ---------------------------------------------------------------------------
### --------------- FUNCTION DEFINITION ------------------------------------


# Function to Check if the video was opened successfully
def check_video_opened(cap, video_path):
    if not cap.isOpened():
        print("Error: Could not open Video")
    else:
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video Loaded: {video_path}")
        print(f"Resolution: {width} x {height}")
        print(f"Frame Rate: {fps} FPS")
        print(f"Total Frames: {total_frames}")
        print(f"Total Sampled Frames: 10")
    cap.release()
    
# Function to get frame count
def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

# Function to sample frames uniformly
def sample_frames_uniformly(video_path, num_samples, ref_mean, ref_std, GC, gamma):
    frame_count = get_frame_count(video_path)
    interval = frame_count // num_samples
    frames = []
    
    cap = cv2.VideoCapture(video_path)
    for i in range(0, frame_count, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        frame = frame.astype('float32')
        if GC==True:
            # Apply gamma correction
            standardized_frame = cv2.pow(frame, gamma)
            # Scale back to range 0-255 and convert to uint8
            # standardized_frame = np.uint8(standardized_frame*255)
            # Getting mean and standard deviation value
            # ref_mean, ref_std = cv2.meanStdDev(standardized_frame)
            # print("ref_mean", ref_mean)
            # print("ref_std", ref_std)
        else:
            # Standardize the frame using provided mean and standard deviation
            standardized_frame = (frame - ref_mean*255) / (ref_std*255)
        if ret:
            frames.append(standardized_frame)
        else:
            break
    cap.release()
    return frames[0:num_samples]

# Function to sample frames randomly
def sample_frames_randomly(video_path, num_samples, ref_mean, ref_std, GC, gamma):
    frame_count = get_frame_count(video_path)
    frames = []
    
    # Generate num_samples random unique frame indices
    random_indices = random.sample(range(frame_count), num_samples)
    
    cap = cv2.VideoCapture(video_path)
    for idx in sorted(random_indices):  # Sort the indices to maintain temporal order
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frame = frame.astype('float32')
        if GC==True:
            # Apply gamma correction
            standardized_frame = cv2.pow(frame, gamma)
            # Scale back to range 0-255 and convert to uint8
            # standardized_frame = np.uint8(standardized_frame*255)
            # Getting mean and standard deviation value
            # ref_mean, ref_std = cv2.meanStdDev(standardized_frame)
            # print("ref_mean", ref_mean)
            # print("ref_std", ref_std)
        else:
            # Standardize the frame using provided mean and standard deviation
            standardized_frame = (frame - ref_mean*255) / (ref_std*255)
        if ret:
            frames.append(standardized_frame)
        else:
            print(f"Frame at index {idx} could not be read")
            continue  # Continue to try to read the next frame if this one fails
    cap.release()
    return frames[0:num_samples]

# Function to display frames using matplotlib
def display_frames(frames):
    fig, axes = plt.subplots(1, len(frames), figsize=(15, 5))
    for ax, frame in zip(axes, frames):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis('off')
    plt.show()
    
# Function to extract feature from frames using ResNet50
def extract_features(frames, model):
    features = np.empty((1,0))
    idx = 0;
    while idx < len(frames):
        # Resize the frame to 224 x 224 pixels, assumed that it is already in float32
        frame = cv2.resize(frames[idx], (224, 224))
        # Add an extra dimension (for batch size)
        frame = np.expand_dims(frame, axis=0)
        # Preprocess the frame for ResNet50
        frame = preprocess_input(frame)
        # Get the features for the frame
        feature = model.predict(frame)
        # flatten the features\
        feature = feature.flatten()
        # Append array of features
        features = np.append(features, feature)
        idx += 1
    return features

# Function to get sampled, normalized frames from train folder
def get_train_frames(train_path, num_samples, ref_mean, ref_std, GC, gamma):
    frames = []
    labels = []
    act = ['Jump', 'Run', 'Sit', 'Stand', 'Turn', 'Walk']
    idx = 0
    while idx < len(act):
        videos_path = train_path + "/" + act[idx]
        # print(videos_path)
        for path in glob.glob(os.path.join(videos_path, "*.mp4")):
            print(path)
            cap = cv2.VideoCapture(path)
            # check_video_opened(cap, path)
            frame = sample_frames_uniformly(path, num_samples, ref_mean, ref_std, GC, gamma)
            frames.append(frame)
            labels.append(idx)
        idx += 1
    return frames, labels

# Function to get sampled, normalized frames from validate folder
def get_validate_frames(validate_path, num_samples, ref_mean, ref_std, GC, gamma):
    frames = []
    labels = [0]*17 + [1]*15 + [2]*15 + [3]*16 + [4]*17 + [5]*16
    for path in glob.glob(os.path.join(validate_path, "*.mp4")):
        # print(path)
        cap = cv2.VideoCapture(path)
        # check_video_opened(cap, path)
        frame = sample_frames_uniformly(path, num_samples, ref_mean, ref_std, GC, gamma)
        frames.append(frame)
    return frames, labels

# Function to get features from train folder
def get_train_features(train_path, num_samples, model, ref_mean, ref_std, GC, gamma):
    features = np.empty((0, 2048*num_samples))
    labels = np.empty((1,0), dtype=np.int32)
    act = ['Jump', 'Run', 'Sit', 'Stand', 'Turn', 'Walk']
    idx = 0
    while idx < len(act):
        videos_path = train_path + "/" + act[idx]
        # print(videos_path)
        for path in glob.glob(os.path.join(videos_path, "*.mp4")):
            # print(path)
            cap = cv2.VideoCapture(path)
            # check_video_opened(cap, path)
            frame = sample_frames_uniformly(path, num_samples, ref_mean, ref_std, GC, gamma)
            feature = extract_features(frame, model)
            features = np.vstack((features, feature))
            labels = np.append(labels, idx)
        idx += 1
    return features, labels

# Function to get sampled, normalized frames from validate folder
def get_validate_features(validate_path, num_samples, model, ref_mean, ref_std, GC, gamma):
    features = np.empty((0, 2048*num_samples))
    labels = np.array([0]*17 + [1]*15 + [2]*15 + [3]*16 + [4]*17 + [5]*16)
    for path in glob.glob(os.path.join(validate_path, "*.mp4")):
        # print(path)
        cap = cv2.VideoCapture(path)
        # check_video_opened(cap, path)
        frame = sample_frames_uniformly(path, num_samples, ref_mean, ref_std, GC, gamma)
        feature = extract_features(frame, model)
        print(feature)
        features = np.vstack((features, feature))
    return features, labels

# Function to sample frames for CNN
def sample_frames_CNN(video_path, num_steps, ref_mean, ref_std, GC, gamma):
    frame_count = get_frame_count(video_path)
    interval = frame_count // num_steps
    samples = []
    
    cap = cv2.VideoCapture(video_path)
    for i in range(0, frame_count, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        frame = frame.astype('float32')
        if GC==True:
            # Apply gamma correction
            standardized_frame = cv2.pow(frame, gamma)
            # Scale back to range 0-255 and convert to uint8
            standardized_frame = np.uint8(standardized_frame*255)
            # Getting mean and standard deviation value
            # ref_mean, ref_std = cv2.meanStdDev(standardized_frame)
            # print("ref_mean", ref_mean)
            # print("ref_std", ref_std)
        else:
            # Standardize the frame using provided mean and standard deviation
            standardized_frame = (frame - ref_mean*255) / (ref_std*255)
        standardized_frame = standardized_frame.flatten()
        if ret:
            samples.append(standardized_frame)
        else:
            break
    cap.release()
    samples = np.array(samples[0:num_steps])
    return samples

# Function to get sampled, normalized frames from train folder
def get_train_samples(train_path, num_steps, ref_mean, ref_std, GC, gamma):
    samples = []
    labels = np.empty((1,0), dtype=np.int32)
    act = ['Jump', 'Run', 'Sit', 'Stand', 'Turn', 'Walk']
    idx = 0
    while idx < len(act):
        videos_path = train_path + "/" + act[idx]
        # print(videos_path)
        for path in glob.glob(os.path.join(videos_path, "*.mp4")):
            # print(path)
            cap = cv2.VideoCapture(path)
            # check_video_opened(cap, path)
            sample = sample_frames_CNN(path, num_steps, ref_mean, ref_std, GC, gamma)
            samples.append(sample)
            labels = np.append(labels, idx)
        idx += 1
    samples = np.array(samples)
    return samples, labels

# Function to get sampled, normalized frames from validate folder
def get_validate_samples(validate_path, num_steps, ref_mean, ref_std, GC, gamma):
    samples = []
    labels = np.array([0]*17 + [1]*15 + [2]*15 + [3]*16 + [4]*17 + [5]*16)
    for path in glob.glob(os.path.join(validate_path, "*.mp4")):
        # print(path)
        cap = cv2.VideoCapture(path)
        # check_video_opened(cap, path)
        sample = sample_frames_CNN(path, num_steps, ref_mean, ref_std, GC, gamma)
        samples.append(sample)
    samples = np.array(samples)
    return samples, labels


### ---------------------------------------------------------------------------
### ------------------------- Main Program  -----------------------------------

### ----------------------- Check Samplings of one video clips
video_path = 'train/Jump/Jump_8_1.mp4'

## Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

## Define some variables for normalization
ref0_mean = np.array([0, 0, 0], dtype='float32')
ref0_std = np.array([1, 1, 1], dtype='float32')
ref1_mean = np.array([0.07, 0.07, 0.07], dtype='float32')
ref1_std = np.array([0.1, 0.09, 0.08], dtype='float32')
enh_mean = np.array([0.485, 0.456, 0.406], dtype='float32')
enh_std = np.array([0.229, 0.224, 0.225], dtype='float32')

## Define the feature extraction
model = ResNet50(include_top=False, weights='imagenet', pooling='avg')

## Check if the video was opened successfully
check_video_opened(cap, video_path)

## Getting sampled frames (with normalization)
frames_0 = sample_frames_uniformly(video_path, 10, ref0_mean, ref0_std, GC=False, gamma=0)
frames_1 = sample_frames_uniformly(video_path, 10, ref1_mean, ref1_std, GC=False, gamma=0)
frames_enh = sample_frames_uniformly(video_path, 10, enh_mean, enh_std, GC=False, gamma=0)
frames_gc = sample_frames_uniformly(video_path, 10, ref0_mean, ref0_std, GC=True, gamma = 0.12)

# # Display sampled frames
display_frames(frames_0)
display_frames(frames_1)
display_frames(frames_enh)
display_frames(frames_gc)

### Extract one Feature from one Frame using defined model
features = extract_features(frames_gc, model)

### --------------------------------------------------------------------------
### ------------------------ SVM for HAR -------------------------------------

###  Get features from All Videos (for Kernel SVM)
(first Normalization Setting)
features_tr0, labels_tr0 = get_train_features('train', 10, model, ref0_mean, ref0_std, GC=False, gamma=0)
features_val0, labels_val0 = get_validate_features('validate', 10, model, ref0_mean, ref0_std, GC=False, gamma=0)

## (second Normalization Setting)
features_tr1, labels_tr1 = get_train_features('train', 10, model, ref1_mean, ref1_std, GC=False, gamma=0)
features_val1, labels_val1 = get_validate_features('validate', 10, model, ref1_mean, ref1_std, GC=False, gamma=0)

## (third Normalization Setting)
features_tr_en, labels_tr_en = get_train_features('train', 10, model, enh_mean, enh_std, GC=False, gamma=0)
features_val_en, labels_val_en = get_validate_features('validate', 10, model, enh_mean, enh_std, GC=False, gamma=0)

### (fourth Normalization Setting)
features_tr_gc, labels_tr_gc = get_train_features('train', 10, model, ref0_mean, ref0_std, GC=True, gamma=0.12)
features_val_gc, labels_val_gc = get_validate_features('validate', 10, model, ref0_mean, ref0_std, GC=True, gamma=0.12)

# Apply Kernel SVM Classifier into the obtained feature 
svm_with_kernel = SVC(gamma=0.01, kernel='rbf', probability=True)
svm_with_kernel.fit(features_tr_gc, labels_tr_gc) 
y_pred = svm_with_kernel.predict(features_val_gc)
precision = metrics.accuracy_score(y_pred, labels_val_gc) * 100
print("Accuracy of Kernel SVM: {0:.2f}%".format(precision))

# Using Kernel SVM Classifier into the obtained feature with PCA
pca = PCA(n_components = 2)
features_tr_gc = pca.fit_transform(features_tr_gc)
features_val_gc = pca.fit_transform(features_val_gc)
svm_with_kernel.fit(features_tr_gc, labels_tr_gc)
y_pred = svm_with_kernel.predict(features_val_gc)
precision = metrics.accuracy_score(y_pred, labels_val_gc) * 100
print("Accuracy of Kernel SVM with PCA: {0:.2f}%".format(precision))

# Plotting decision boundaries
plot_decision_regions(features_tr_gc, labels_tr_gc, clf=svm_with_kernel, legend=1)
plt.title('Kernel SVM Decision Boundaries')
plt.show()


### --------------------------------------------------------------------------
### ------------------------ Late Fusion of RNN and LSTM ---------------------

# Load the training and validate frames
frames_train, labels_train = get_train_frames('train', 10)
frames_validate, labels_validate = get_validate_frames('validate', 10)

### Load the training and validate sampled frames (first Normalization Setting)
samples_tr0, labels_tr0 = get_train_samples('train', 10, ref0_mean, ref0_std, GC=False, gamma=0)
samples_val0, labels_val0 = get_validate_samples('validate', 10, ref0_mean, ref0_std, GC=False, gamma=0)

### Load the training and validate sampled frames (second Normalization Setting)
samples_tr1, labels_tr1 = get_train_samples('train', 10, ref1_mean, ref1_std, GC=False, gamma=0)
samples_val1, labels_val1 = get_validate_samples('validate', 10, ref1_mean, ref1_std, GC=False, gamma=0)

### Load the training and validate sampled frames (third Normalization Setting)
samples_tr_en, labels_tr_en = get_train_samples('train', 10, enh_mean, enh_std, GC=False, gamma=0)
samples_val_en, labels_val_en = get_validate_samples('validate', 10, enh_mean, enh_std, GC=False, gamma=0)

### Load the training and validate sampled frames (fourth Normalization Setting)
samples_tr_gc, labels_tr_gc = get_train_samples('train', 10, ref0_mean, ref0_std, GC=True, gamma=0.12)
samples_val_gc, labels_val_gc = get_validate_samples('validate', 10, ref0_mean, ref0_std, GC=True, gamma=0.12)

# One-hot encode the labels
y_one_hot = to_categorical(labels_tr_gc, 6)

# Define the RNN model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(50, input_shape=(samples_tr_gc.shape[1], samples_tr_gc.shape[2]), return_sequences=False))
rnn_model.add(Dense(y_one_hot.shape[1], activation='softmax'))

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(samples_tr_gc.shape[1], samples_tr_gc.shape[2]), return_sequences=False))
lstm_model.add(Dense(y_one_hot.shape[1], activation='softmax'))

# Compile the models
rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Get predictions from both models
rnn_predictions = rnn_model.predict(samples_val_gc)
lstm_predictions = lstm_model.predict(samples_val_gc)

# Late fusion: here we simply average the predictions
fused_predictions = (rnn_predictions + lstm_predictions) / 2.0

# Convert predictions to actual labels
final_predictions = np.argmax(fused_predictions, axis=1)

# Evaluate the late fusion model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_val_gc, final_predictions)
print(f'Late Fusion Accuracy: {accuracy * 100:.2f}%')