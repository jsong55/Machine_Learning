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
