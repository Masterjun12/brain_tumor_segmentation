import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def visual_num(n, test_images, test_labels, model):
    num =n
    test_image = test_images[num]
    test_label = test_labels[num]
    prediction = model.predict(np.expand_dims(test_image, axis=0))[0]
    prediction = np.argmax(prediction, axis=-1)
# 예측 결과 시각화
    plt.figure(figsize=(9,5))
    plt.subplot(241) ;  plt.imshow(test_image[:,:,0], cmap='gray') ; plt.title('T1');plt.axis('off')
    plt.subplot(242) ;  plt.imshow(test_image[:,:,1], cmap='gray') ; plt.title('T1ce'); plt.axis('off')
    plt.subplot(243) ;  plt.imshow(test_image[:,:,2], cmap='gray') ; plt.title('T2'); plt.axis('off')
    plt.subplot(244) ;  plt.imshow(test_image[:,:,3], cmap='gray') ; plt.title('flair'); plt.axis('off')
    t1 = np.where(test_label == 0, 1,0).astype('uint8')
    t2 = np.where(test_label == 1, 1,0).astype('uint8')
    t3 = np.where(test_label == 2, 1,0).astype('uint8')
    t4 = np.where(test_label == 3, 1,0).astype('uint8')
    plt.subplot(245); plt.imshow(test_label, cmap='gray'); plt.title('Ground Truth');plt.axis('off')
    plt.subplot(2,4,6) ; plt.imshow(prediction, cmap='gray'); plt.title('Predicted');plt.axis('off')
    plt.subplot(2,4,7) ; plt.imshow(abs(test_label-prediction), cmap='gray'); plt.title('GT - pred');plt.axis('off')
    plt.subplot(2,4,8) ; plt.imshow(test_image[:,:,3], cmap='gray'); plt.title('flair overlab');plt.axis('off')
    plt.subplot(2,4,8) ; plt.imshow(abs(test_label-prediction), cmap='jet',alpha=0.5);plt.axis('off')
    t = test_label.flatten()
    p = prediction.flatten()
    matches = t == p
    num_mismatches = len(matches) - np.count_nonzero(matches)
    mismatch_indices = np.where(matches == False)[0]
    print("mismatched labels:", num_mismatches)
    plt.tight_layout();plt.show()