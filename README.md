# brain-tumor-Segmentation

## Contents

  - [Overview](#overview)
  - [Basic Concepts](#basic-concepts)
    + [Sementic Segmentation](#sementic-segmentation)
    + [Atrous Convolution(Dilated Convolution)](#atrous-convolution-dilated-convolution-)
    + [Spatial Pyramid Pooling](#spatial-pyramid-pooling)
    + [U-net](#u-net)
  - [Environment](#environment)
    + [research environment](#research-environment)
  - [DataSet](#dataset)
    + [EDA and Data Processing](#eda-and-data-processing)
  - [Modeling and Fit](#modeling-and-fit)
    + [Base Segmentation](#base-segmentation)
  - [Experiment](#experiment)
    + [Experiment results](#experiment-results)
  - [References](#references)

---

## Overview


The Multimodal Brain Tumor Segmentation Challenge, particularly utilizing MRI brain tumor data from [**Brain Tumor Segmentation (BraTS) Challenge 2020: Scope](https://www.med.upenn.edu/cbica/brats2020/),** has spurred extensive research endeavors aimed at advancing the precision of semantic segmentation models for brain tumors.

Researchers have embarked on various attempts to refine brain tumor segmentation, starting with fundamental architectures like U-Net and progressing to incorporate a myriad of convolutional methodologies and insights gleaned from reference papers. The primary focus has been to address the inherent limitations of existing segmentation models, particularly in accurately delineating tumor boundaries.

Building upon the foundation laid by U-Net, these efforts have explored innovative convolutional techniques and integrated novel ideas gleaned from scholarly works. The overarching objective has been to enhance the prediction of tumor margins, thereby striving towards achieving a more comprehensive and precise methodology for brain tumor segmentation.

Through meticulous experimentation and iterative refinement, researchers have sought to push the boundaries of existing segmentation techniques, with the ultimate aim of facilitating more accurate diagnosis and treatment planning for patients with brain tumors. This journey underscores the collective pursuit within the research community towards continually refining and perfecting the art of brain tumor segmentation.

## Basic Concepts

---

### Sementic Segmentation

- 실제로 인식할 수 있는 물리적 의미 단위로 인식하는 세그멘테이션을 시멘틱 세그멘테이션(sementic segmentation)

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled.png)

- 이미지를 이루는 모든 픽셀이 레이블을 보유하고 있음

### Atrous Convolution(Dilated Convolution)

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%201.png)

Atrous Convolution은 r(rate)라는 파라미터를 통해 얼마나 빈 공간을 둘지 설정하고 빈 공간을 제외하고 기존 Convolution과 같이 학습을 진행

r = 1일 경우에는 기존 Convolution과 동일하다는 뜻

장점

1. field of view(한 픽셀이 볼 수 있는 영역)를 크게 가져갈 수 있으면서 일반 컨볼루션과 동일한 연산량
2. Receptive Field 를 늘려서 필터가 볼 수 있는 영역을 확대한다

### Spatial Pyramid Pooling

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%202.png)

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%203.png)

① 각 feature map은 윈도우 크기가 1x1으로 풀링 된 256-d vector가 생성(가장 오른쪽 회색)

② 각 feature map은윈도우 크기가 2x2로 풀링 된 4x256-d vector가 생성.(중간 초록색)

③ 각 feature map은 윈도우 크기가 4x4으로 풀링 된 16x256-d vector가 생성(가장 왼쪽 파란색)

④ 위의 3개의 vector를 concat하여 1-d vector를 생성

⑤ 이 vector를 FC layer로 전달

### U-net

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%204.png)

유넷(U-Net)은 주로 의료 이미지 분석을 위해 개발된 딥러닝 아키텍처
의료 영상에서 특정 구조물 또는 병변을 정확하게 분할하기 위한 목적으로 고안되었음

1. **인코더-디코더 구조**: 유넷은 대칭적이고 인코더(encoder)와 디코더(decoder)라는 두 가지 주요 구성 요소로 이루어짐. 인코더는 입력 이미지를 점진적으로 다운샘플링하여 특성을 추출, 디코더는 이러한 특성을 업샘플링하여 입력 이미지와 동일한 해상도로 복원
2. **스킵 연결(skip connections)**: 인코더와 디코더 간의 스킵 연결. 이는 인코더의 각 단계에서 추출된 특성 맵을 디코더의 해당 단계로 직접 연결하여 정보 유실을 방지, 더 정확한 분할가능
3. **잔여 연결(residual connections)**: 일부 유넷 구조에서는 잔여 연결이 사용됨. 이는 디코더의 각 층에서 이전 층의 출력을 현재 층의 입력에 추가하여 정보의 흐름을 최적화

### Dice Loss

![Untitled](https://github.com/Masterjun12/brain_tumor_segmentation/blob/6dc6fa98fe07e4419fdfd1b84fdfe51729cbff6c/brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%2019.png)

Dice loss는 세그멘테이션 문제에 널리 사용되는 손실 함수
Dice loss는 Dice Similarity Coefficient (DSC)측정 지표를 기반함

두 샘플 집합 간의 유사도를 측정
이를 로스펑션으로 하여 두 집합 샘플간의 교집합이 높아지게 하는것이 목적


## Environment

---

### research environment

```markdown
hardwear : Nvidia A100

cuda : version 12.2
Ubuntu 22.04.1 LTS
python : 3.8.14
tensorflow : 2.13.0
```

## DataSet

---

- DataSet : BRATS 2017

- MRI 채널 설명:

- T1 (T1-weighted):
  고해상도로 뇌 구조를 시각화하며, 뇌의 해부학적 정보를 잘 보여줌.
  CSF(뇌척수액)가 어둡게 나타나는 특징.
- T1ce (T1 with contrast enhancement):
  조영제를 주입한 후 촬영된 T1 이미지.
  종양 부위가 강조되어 종양의 경계와 특징을 명확히 파악 가능.
- T2 (T2-weighted):
  뇌의 수분 함량을 잘 반영하며, 병변과 종양 주변의 부종 영역을 확인하는 데 유용.
  CSF가 밝게 나타나는 특징.
- FLAIR (Fluid-Attenuated Inversion Recovery):
  T2 기반 이미지로, CSF를 억제하여 병변과 종양 영역을 더 선명하게 표현.
  비정상 조직(예: 부종)을 감지하는 데 유용.

- image
    - mri는 4가지 양식(t1,t1ce,t2,flair)으로 촬영되었고, 4개를 하나의 데이터(RGB Image가 3channel인것과 같음)로 사용함
    
    ![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%205.png)
    
- label
    - 0 : 배경
    - 1 : 전체 종양(Whole tumor)
    - 2 : 핵심 종양(Tumor core)
    - 3 : 조영제 등으로 인해 강조된 종양(enhancing tumor)
    

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%206.png)

```
train_images : (3906, 240, 240, 4)
train_labels : (3906, 240, 240)
val_images   : (1302, 240, 240, 4)
val_labels   : (1302, 240, 240)
test_images  : (1302, 240, 240, 4)
test_labels  : (1302, 240, 240)
```

### EDA and Data Processing

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%207.png)

image, label overlab

이런식으로 랜덤하게 5장 더 확인해 본다

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%208.png)

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%209.png)

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%2010.png)

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%2011.png)

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%2012.png)

- 각각히 뇌모양도 종양의 모양과 크기도다르다는 것을 알수 있다.
- 기준에 따른 전처리가 요구된다.

데이터 셋을 생성하기전 고려해야 할 중요한 사항이 있다

Mri 사진 데이터는 입체적인 뇌를 분할 하여 촬영 하였기 때문에

![MRI 촬영을 설명하기 위한 예시](https://github.com/Masterjun12/brain_tumor_segmentation/blob/6dc6fa98fe07e4419fdfd1b84fdfe51729cbff6c/brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%2013.gif)

MRI 촬영을 설명하기 위한 예시

종양의 아래 부분이나, 윗 부분의 종양의 크기는 매우 작을 것이고 

이는 학습에 방해 요인이 될 수 도 있다.

![예시로 이러한 데이터 셋이 있을 수 있다.](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%208.png)

예시로 이러한 데이터 셋이 있을 수 있다.

우리는 종양의 중심부라고 생각되고 레이블의 면적이 일정 범위 이상인 데이터를 선정하기로 한다

```
Label 0 pixels: 367211293
Label 1 pixels: 1140885
Label 2 pixels: 5163680
Label 3 pixels: 1460142
```

배경이 0번이 매우 많고 2번이 많다

```
Number of images with no label 0 pixels: 0
Number of images with no label 1 pixels: 2850
Number of images with no label 2 pixels: 60
Number of images with no label 3 pixels: 2617
```

1번과 3번의 레이블이 없는 이미지들이 다수 존재한다

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%2014.png)

2번 레이블은 1000개 이상의 픽셀을 가진 이미지가 많고 200~300개 픽셀 범위는 1,3 에 비해 매우 적다

그말은 핵심종양의 크기가 크다는 것을 알 수가 있다

## Modeling and Fit

---

### Base Segmentation

```markdown
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, Dropout
from tensorflow.keras.models import Model

input_shape = (240, 240, 4)
inputs = Input(shape=input_shape)

# Encoder
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2) 
conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

# Middle
conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4) 
conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

# Decoder
up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=-1)
conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=-1)
conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=-1)
conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=-1)
conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

# Output
outputs = Conv2D(4, (1, 1), activation='softmax')(conv9)
model = Model(inputs=inputs, outputs=outputs)
model.summary()

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

model.compile(optimizer='adam', loss=bce_dice_loss, metrics=['accuracy'])

history= model.fit(train_images, train_labels, epochs=100, batch_size=32, validation_data=(val_images, val_labels))
```

- 기본적인 Unet 모델을 통하여 성능을 확인해 보았다
- 옵티마이저는 adam
- 로스함수는 bce_dice_loss(Binary Cross Entropy and Dice Loss)

### 결과

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%2015.png)

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%2016.png)

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%2017.png)

![Untitled](brain-tumor-Segmentation%209669ed4be1ac4f33b8731e4ff2c8b195/Untitled%2018.png)

- 위 5가지 이미지들은 랜덤한 Test 이미지 예측결과이다.
- 공통적으로 뇌종양의 테두리 부분을 인식하지 못하는 문제가 있다
- 이 부분을 모델,로스,전처리 등 다양한 방법을 통해 개선할 것 이다.

|  | Sensitivity | Specificity | F1 Score | Precision |
| --- | --- | --- | --- | --- |
| 0 | 0.99840 | 0.9567 | 0.998700 | 0.999100 |
| 1 | 0.76060 | 0.9998 | 0.834500 | 0.924400 |
| 2 | 0.92060 | 0.9985 | 0.907100 | 0.893900 |
| 3 | 0.94500 | 0.9990 | 0.857800 | 0.785300 |
| Col Ave | 0.90615 | 0.9885 | 0.899525 | 0.900675 |
- 선택한 measure 로는 recall 값을 이용한다.

## Experiment

---

### Experiment results

- 모델의 변형
- 로스의 변형
- 데이터셋 전처리

---

### Methods

- **Cropped**  
  Cropping the image to reduce the background area, expecting performance improvement.  
  ![Cropped](brain-tumor-Segmentation/9669ed4be1ac4f33b8731e4ff2c8b195/croped.jpg)  

- **Gaussian Blur**  
  Applying Gaussian blur based on referenced research papers.  
  ![Gaussian Blur](brain-tumor-Segmentation/9669ed4be1ac4f33b8731e4ff2c8b195/가우시안.jpg)  

- **Edge Map**  
  Extracting the edge of the actual brain tumor mask in the training dataset and synthesizing it with the tumor image. Used in the edge enhancement model.  
  ![Edge Map](brain-tumor-Segmentation/9669ed4be1ac4f33b8731e4ff2c8b195/엣지맵.jpg)  
  ![Edge Synthesis](brain-tumor-Segmentation/9669ed4be1ac4f33b8731e4ff2c8b195/엣지합성.jpg)  

- **Autoencoder Generation**  
  Training an autoencoder to generate images by synthesizing the brain tumor mask. After training, the model generates images for validation and test datasets, using them as input for further processing.  
  ![Autoencoder](brain-tumor-Segmentation/9669ed4be1ac4f33b8731e4ff2c8b195/오토인코더.jpg)  

- **Clustering**  
  Using the K-means algorithm with K=5, assuming that the middle cluster represents the brain tumor mask. This preprocessing is based on the hypothesis that the central feature value of each brain tumor image corresponds to the tumor mask.  
  ![Clustering](brain-tumor-Segmentation/9669ed4be1ac4f33b8731e4ff2c8b195/클러스터링.jpg)  


### Model Performance Overview

The table below summarizes the performance of various models trained on different datasets with different loss functions and optimizers. Metrics such as Recall, F1 Score, Precision, and Specificity are used to evaluate the models.

| **Title**                               | **Base Model**   | **Dataset**  | **Parameter (Size)**    | **Optimizer**  | **Loss Function**               | **Recall** | **F1 Score** | **Precision** | **Specificity** |
|-----------------------------------------|------------------|--------------|-------------------------|----------------|----------------------------------|------------|--------------|---------------|-----------------|
| U-Net                                   | U-net            | Original     | 31032516 (118.38 MB)     | Adam           | Bce_Dice_Loss                   | 0.914275   | 0.91997      | 0.924925      | 0.984175        |
| Attention U-Net                         | Attention U-Net  | Original     | 31870853 (121.58 MB)     | Adam           | Bce_Dice_Loss                   | 0.926376   | 0.92889      | 0.931425      | 0.988425        |
| Res-U-Net++                             | Res-Unet++       | Original     | 4092868 (15.61 MB)       | Adam           | Bce_Dice_Loss                   | 0.911125   | 0.92221      | 0.933575      | 0.983725        |
| U-Net (3++)                             | U-net 3++        | Original     | 8907396 (33.98 MB)      | Adam           | Bce_Dice_Loss                   | 0.92425    | 0.92335      | 0.922425      | 0.986125        |
| 3++ Unet Nobridge Ell-Loss             | U-net 3++        | Original     | 8907396 (33.98 MB)      | Adam           | Ell_Loss (0.2)                  | 0.92805    | 0.92965      | 0.93135       | 0.988625        |
| U-net Nobridge                          | U-net            | Original     | 15826116 (60.37 MB)     | Adam           | Bce_Dice_Loss                   | 0.92415    | 0.92325      | 0.922525      | 0.98885         |
| Res-Unet++ Nobridge                    | Res-Unet++       | Original     | 4092868 (15.61 MB)      | Adam           | Bce_Dice_Loss                   | 0.912525   | 0.922825     | 0.933575      | 0.985125        |
| U-net Ell-Loss Nobridge                | U-net            | Original     | 15826116 (60.37 MB)     | Adam           | Ell_Loss (0.7)                  | 0.92415    | 0.92325      | 0.922525      | 0.98885         |
| 3Conv Block Unet Ell-Loss              | U-net            | Original     | 46774852 (178.43 MB)    | Adam           | Ell_Loss (0.7)                  | 0.90615    | 0.899525     | 0.900675      | 0.9885          |
| 3Conv Block Unet-Nobridge Ell-Loss     | U-net            | Original     | 22117956 (84.37 MB)     | Adam           | Ell_Loss (0.7)                  | 0.917575   | 0.91495      | 0.912425      | 0.98785         |
| 3Conv Block Unet-Nobridge-Weightedloss | U-net            | Original     | 22117956 (84.37 MB)     | Adam           | Weighted_Categorical_Crossentropy | 0.912925 | 0.909575     | 0.908725      | 0.9791          |
| 3Conv Block Unet-5channel Ell-Loss     | U-net            | 5channel     | 46775428 (178.43 MB)    | Adam           | Ell_Loss (0.7)                  | 0.90575    | 0.916925     | 0.929375      | 0.98385         |
| 3Conv Block Unet-5channel-Nobridge     | U-net            | 5channel     | 22118532 (84.38 MB)     | Adam           | Bce_Dice_Loss                   | 0.9115     | 0.916575     | 0.921925      | 0.982225        |
| 3Conv Block Unet-5channel-Nobridge_wloss | U-net          | 5channel     | 18127940 (69.15 MB)     | Adam           | Weighted_Categorical_Crossentropy | 0.922925 | 0.92735      | 0.932         | 0.985725        |
| 3Conv Block Unet-5channel-Nobridge     | U-net            | 5channel     | 12046212 (45.95 MB)     | Adam           | Bce_Dice_Loss                   | 0.9264     | 0.9286       | 0.93115       | 0.988825        |
| 3Conv_Block_3++_Softdice               | U-net 3++        | Original     | 12046212 (45.95 MB)     | Adagrad        | Soft_Dice_Loss                  | 0.870175   | 0.887325     | 0.9056        | 0.975275        |
| Unet_Dropout                           | U-net            | Original     | 31032516 (118.38 MB)    | Adam           | Bce_Dice_Loss                   | 0.9253     | 0.92805      | 0.931025      | 0.986175        |
| Unet_Nobridge_Dropout                 | U-net            | Original     | 15826116 (60.37 MB)     | Adagrad        | Bce_Dice_Loss                   | 0.656725   | 0.657225     | 0.7663        | 0.9511          |
| Unet_Dropout_Soft_Softdice            | U-net            | Original     | 31032516 (118.38 MB)    | Adagrad        | Soft_Dice_Loss                  | 0.607625   | 0.64925      | 0.800775      | 0.918125        |
| Res-Unet++-Dropout_Softdice           | Res-Unet++       | Original     | 4092868 (15.61 MB)      | Adagrad        | Soft_Dice_Loss                  | 0.772875   | 0.78815      | 0.8222        | 0.97215         |
| Unet_Dropout-Dilation_Rate           | U-net            | Original     | 31379716 (119.70 MB)    | Adam           | Bce_Dice_Loss                   | 0.9169     | 0.92425      | 0.9317        | 0.9846          |
| Unet_Dilation_Rate                    | U-net            | Original     | 31379716 (119.70 MB)    | Adam           | Bce_Dice_Loss                   | 0.926875   | 0.9262       | 0.925725      | 0.98855         |
| 3++_Nobridge-Dropout-E3              | U-net 3++        | Original     | 8907396 (33.98 MB)      | Adam           | Bce_Dice_Loss                   | 0.921925   | 0.928925     | 0.936225      | 0.984475        |
| 3Conv_Block_3++_5Channel-Nobridge_Wloss-Ell-Dilation_Rate=3 | U-net 3++ | Original | 15911684 (60.70 MB) | Adam | Weighted_Categorical_Crossentropy | 0.926275 | 0.923475 | 0.9214 | 0.9866 |
| Unet_Dropout-Dilation_Rate_Nobridge | U-net | Original | 12046212 (45.95 MB) | Adam | Bce_Dice_Loss | 0.9117 | 0.924175 | 0.937175 | 0.982825 |
| Cropped_4Channel_Unet_Nobridge | U-net | Cropped | 15826116 (60.37 MB) | Adam | Bce_Dice_Loss | 0.7519 | 0.787425 | 0.83205 | 0.94565 |
| Attention_Unet / Categorical_Crossentropy | Attention U-Net | Original | 31870853 (121.58 MB) | Adam | Categorical_Crossentropy | 0.926376 | 0.929 | 0.931775 | 0.9876 |
| Attention_Unet / Bce_Dice_Loss       | Attention U-Net  | Original     | 31870853 (121.58 MB)     | Adam           | Bce_Dice_Loss                   | 0.929      | 0.93         | 0.9311        | 0.98895         |
| Attention_Unet / Weighted_Categorical_Crossentropy | Attention U-Net | Original | 31870853 (121.58 MB) | Adam | Weighted_Categorical_Crossentropy | 0.929625 | 0.925325 | 0.921325 | 0.987025 |
| Attention_Unet / Weighted_Categorical_Crossentropy (Bayesian Optimization) | Attention U-Net | Original | 31870853 (121.58 MB) | Adam | Weighted_Categorical_Crossentropy | 0.92375 | 0.900625 | 0.88125 | 0.993175 |
| 3Chan_Unet | U-net | 3channel | 31031940 (118.38 MB) | Adam | Bce_Dice_Loss | 0.9125 | 0.9185 | 0.924775 | 0.982975 |
| Edge Reinforcement Transfer Learning | U-net (twice)    | Edgy Dataset | 31031940 (118.38 MB)    | Adam           | Bce_Dice_Loss                   | 0.892225   | 0.89685      | 0.9026        | 0.979225        |

---


## References

---
