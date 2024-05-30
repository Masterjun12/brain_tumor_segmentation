import numpy as np
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, train_path, val_path, test_path):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
        self.test_images = None
        self.test_labels = None

    def load_and_process_data(self):
        # Load data
        self.train_images = np.load(f'{self.train_path}/train_images.npy')
        self.val_images = np.load(f'{self.val_path}/val_images.npy')
        self.test_images = np.load(f'{self.test_path}/test_images.npy')

        self.train_labels = np.load(f'{self.train_path}/train_labels.npy')
        self.val_labels = np.load(f'{self.val_path}/val_labels.npy')
        self.test_labels = np.load(f'{self.test_path}/test_labels.npy')
        
        # Process labels: change label 4 to 3
        self.train_labels = np.where(self.train_labels == 4, 3, self.train_labels)
        self.val_labels = np.where(self.val_labels == 4, 3, self.val_labels)
        self.test_labels = np.where(self.test_labels == 4, 3, self.test_labels)

        # Shuffle datasets
        train_indices = np.random.permutation(self.train_images.shape[0])
        val_indices = np.random.permutation(self.val_images.shape[0])
        test_indices = np.random.permutation(self.test_images.shape[0])

        self.train_images = self.train_images[train_indices]
        self.train_labels = self.train_labels[train_indices]
        self.val_images = self.val_images[val_indices]
        self.val_labels = self.val_labels[val_indices]
        self.test_images = self.test_images[test_indices]
        self.test_labels = self.test_labels[test_indices]

    def print_shapes(self):
        print("train_images :", self.train_images.shape)
        print("train_labels :", self.train_labels.shape)
        print("val_images   :", self.val_images.shape)
        print("val_labels   :", self.val_labels.shape)
        print("test_images  :", self.test_images.shape)
        print("test_labels  :", self.test_labels.shape)

    def visualize_image(self, num=0):
        plt.figure(figsize=(8, 3))
        plt.subplot(131)
        plt.imshow(self.test_images[num, :, :, 3])
        plt.title("image")
        plt.axis("off")
        plt.subplot(132)
        plt.imshow(self.test_labels[num, :, :], cmap='gray')
        plt.title("label")
        plt.axis("off")
        plt.subplot(133)
        plt.imshow(self.test_labels[num, :, :], cmap='gray')
        plt.title("overlab")
        plt.axis("off")
        plt.imshow(self.test_images[num, :, :, 3], alpha=0.5)
        plt.axis("off")
        plt.show()