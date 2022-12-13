# Nanowire AF Project

This repository is used to store google colab codes to create ML models to achieve auto-focus and pose estimation for nanowires.

## Progress:

### for 10/24/2022 meeting:
Using the existing image database with several augmentations and pixel reversion, three networks were tested for accuracy:
1. ResNet ("https://arxiv.org/abs/1512.03385"): Using transfer-learning without training parameters, the training accuracy was 35%, validation accuracy was 30%.
2. EfficientNetB0 ("https://arxiv.org/abs/1905.11946"): Using transfer-learning without training parameters, the training accuracy was 72%, validation accuracy was 57%. Using trainable parameters, the training accuracy was 98.9%, validation accuracy was 62.5%. 
3. DenseNet ("https://arxiv.org/abs/1608.06993"): Using trainable parameters, the training accuracy was 95.5%, validation accuracy was 53.1%. 

The training accuracy is hovering around 50%, which could be:
1. Model is too complicated?
2. Need better data.
3. Reduce learning rate.

Overall, using CNN to predict depth during motion is unpredictable. Gather better data: fixed x-y rotation angle or self-generated image data.

* Check out this link ("https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/object_localization_and_detection") for more information
* Check out this link for multi-object identification ("https://aman.ai/cs231n/detection/#multiple-object-detection")

### For 10/31/2022 meeting:
1. After using more and better data, the validation accuracy is approaching 90%.
2. Incline nanowire looks like with tails when it is not focused at the center. Using the AF to track the inclination motion, and incline nanowires look like smaller wires.
3. Figure out the deadline date. feb-1-2023
4. Iron out the outputs of CNNs. (check notebook)
5. Research on more papers.

### For 11/7/2022 meeting:
1. Several experiments were performed to very this: for a tilted wire, the height at the center of the wire has the largest sharpness value, and the histogram of the ROI has a normal distribution shape.
2. The idea of using CNN to learn pose is: use CNN to find the height difference from one tip to its center, and use the projection length to find the angle.
3. The center could be found through AF because it has the highest sharpness value, but how to find two tips.

* Check out this link ("https://towardsdatascience.com/multi-label-image-classification-in-tensorflow-2-0-7d4cf8a4bc72") for multi-label classification, try the example experiments in TF.
* "https://github.com/ashrefm/multi-label-soft-f1/blob/master/Multi-Label%20Image%20Classification%20in%20TensorFlow%202.0.ipynb"

### For 11/28/2022 meeting:


### Overfitting:
The high variance of the model performance is an indicator of an overfitting problem. If the model trains for too long on the training data or is too complex, it learns the noise or irrelevant information within the dataset.

### Underfitting:
The model doesn't perform well for the training dataset.

## Underfitting happens when:
1. Unclean training data containing noise or outliers can be a reason for the model not being able to derive patterns from the dataset.
2. The model has a high bias due to the inability to capture the relationship between the input examples and the target values. 
3. The model is assumed to be too simple. For example, training a linear model in complex scenarios.

### Solve the problem of overfitting by:
1. Increasing the training data by data augmentation
2. Feature selection by choosing the best features and remove the useless/unnecessary features
3. Early stopping the training of deep learning models where the number of epochs is set high
4. Dropout techniques by randomly selecting nodes and removing them from training
5. Reducing the complexity of the model

### Check out this link for batch prediction: "https://gist.github.com/ritiek/5fa903f97eb6487794077cf3a10f4d3e"
and this link "https://www.tensorflow.org/tutorials/keras/classification"

### "https://zhuanlan.zhihu.com/p/465947475" for keras and TF versions.
