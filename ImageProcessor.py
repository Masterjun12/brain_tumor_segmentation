import numpy as np
from sklearn.model_selection import train_test_split

class ImageProcessor:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def select_images(self, threshold):
        selected_images = []
        selected_labels = []
        
        for img, lbl in zip(self.images, self.labels):
            class_1_count = np.sum(lbl == 1)
            class_2_count = np.sum(lbl == 2)
            class_3_count = np.sum(lbl == 3)
            
            if class_1_count > threshold or class_2_count > threshold or class_3_count > threshold:
                selected_images.append(img)
                selected_labels.append(lbl)
        
        selected_images = np.array(selected_images)
        selected_labels = np.array(selected_labels)
        
        # Check if any images were selected
        if len(selected_images) == 0:
            raise ValueError("No images found with class pixel counts exceeding the threshold")
        
        return selected_images, selected_labels
    
    def sample_data(self, fraction):
        _, sampled_images, _, sampled_labels = train_test_split(self.images, self.labels, test_size=fraction, random_state=42)
        return sampled_images, sampled_labels
