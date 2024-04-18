import cv2
import numpy as np
import os

# Parameters
working_dir = r'data'
img_size = 60 # Size of image fed into model

def flatten(dimData, images):
    images = np.array(images)
    images = images.reshape(len(images), dimData)
    images = images.astype('float32')
    images /= 255
    return images

# Get train/test data
folders = ['triangle', 'star', 'square', 'circle']
labels, images = [], []

for folder in folders:
    print(folder)
    for filename in os.listdir(os.path.join(working_dir, 'shapes', folder)):
        filepath = os.path.join(working_dir, 'shapes', folder, filename)
        try:
            img = cv2.imread(filepath, 0)
            if img is not None:
                images.append(cv2.resize(img, (img_size, img_size)))
                labels.append(folders.index(folder))
            else:
                print("Failed to read:", filepath)
        except Exception as e:
            print("Error processing:", filepath)
            print(e)

# Break data into training and test sets
train_images, test_images, train_labels, test_labels = [], [], [], []
to_train = 0

for image, label in zip(images, labels):
    if to_train < 5:
        train_images.append(image)
        train_labels.append(label)
        to_train += 1
    else:
        test_images.append(image)
        test_labels.append(label)
        to_train = 0