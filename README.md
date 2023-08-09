# color_hair
## Introduce
Target of project: change color hair of Person with image

- Use deep learning to determine segmentation mask of hair
- Change color of hair with mask of model

## Solution

- Dataset: Hair Segmentation Dataset (Kaggle)
  + https://www.kaggle.com/datasets/siddhantkulkarni73/hair-segmentation-dataset
- Model: Unet model
  + base: 5
  + input: gray image (128x128x1)
  + output_shape: 128x128

## Train result

![image](https://github.com/khanhngt/color_hair/assets/117704439/be6aafe5-ccab-44f4-8a82-31194ac1790f)

![image](https://github.com/khanhngt/color_hair/assets/117704439/ffe7cb45-5432-492d-9963-939b68d998c8)

- link train with colab: https://colab.research.google.com/drive/1OpZgZ54OCduYMfMfCQOz9eivYi7FALO8?usp=sharing

## Test

- link test with colab: https://colab.research.google.com/drive/10OjPzPyq-buSWF2cyrEbfevP_QwWNH_6?usp=sharing

![image](https://github.com/khanhngt/color_hair/assets/117704439/0cc685f3-3f9c-4d2b-ab52-721f6d8ee6f0)

  
