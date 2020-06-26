# builtin_models

Run `make generate` in the root `pytorch` (`..`) after updating model descriptions.

# model test
### Image Classification

| Name                        | Image                               | Label                  | Probability |
|:---------------------------:|:-----------------------------------:|:----------------------:|:-----------:|
| Caffe_ResNet_101            | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 18.026      |
| DPN_68_v1.0                 | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 20.595      |
| DPN_68_v2.0                 | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 19.059      |
| DPN_92                      | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 18.087      |
| DPN_98                      | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 20.959      |
| DPN_107                     | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 19.373      |
| DPN_131                     | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 22.959      |
| Inception_ResNet_v2.0       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... |  9.391      |
| Inception_v3.0              | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 14.094      |
| NasNet_A_Large              | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... |  9.284      |
| NasNet_A_Mobile             | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... |  9.658      |
| PNasNet_5_Large             | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... |  8.840      |
| PolyNet                     | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 22.046      |
| ResNext101_32x4D            | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 18.788      |
| ResNext101_64x4D            | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 21.068      |
| SE_ResNet_50                | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... |  7.394      |
| SE_ResNet_101               | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... |  7.944      |
| SE_ResNet_152               | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... |  8.439      |
| SE_ResNext_50_32x4D         | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 13.578      |
| SE_ResNext_101_32x4D        | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... |  9.021      |
| SENet_154                   | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... |  8.437      |
| TorchVision_AlexNet         | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 15.774      |
| TorchVision_DenseNet_121    | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 22.642      |
| TorchVision_DenseNet_161    | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 28.404      |
| TorchVision_DenseNet_169    | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 18.400      |
| TorchVision_DenseNet_201    | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 23.355      |
| TorchVision_Resnet_18       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 23.070      |
| TorchVision_Resnet_34       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 19.874      |
| TorchVision_Resnet_50       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 21.993      |
| TorchVision_Resnet_101      | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 20.871      |
| TorchVision_Resnet_152      | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 23.554      |
| TorchVision_SqueezeNet_v1.0 | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 31.827      |
| TorchVision_SqueezeNet_v1.1 | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 29.821      |
| TorchVision_VGG_11          | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 24.545      |
| TorchVision_VGG_11_BN       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 19.582      |
| TorchVision_VGG_13          | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 19.220      |
| TorchVision_VGG_13_BN       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 24.030      |
| TorchVision_VGG_16          | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 19.382      |
| TorchVision_VGG_16_BN       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 22.744      |
| TorchVision_VGG_19          | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 19.382      |
| TorchVision_VGG_19_BN       | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 23.698      |
| Xception                    | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... |  9.881      |

### Image Object Detection

Note: Only recording first five detection, excluding background.
| Name                        | Image                                   | Label | Xmin  | Xmax  | Ymin  | Ymax  | Probability |
|:---------------------------:|:---------------------------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|
| MobileNet_SSD_v1.0          | ../predictor/_fixtures/lane_control.jpg | car   | 0.574 | 0.990 | 0.634 | 1.004 | 0.998       |
|                             |                                         | car   | 0.567 | 0.801 | 0.001 | 0.344 | 0.998       |
|                             |                                         | car   | 0.575 | 0.990 | 0.629 | 0.998 | 0.997       |
|                             |                                         | car   | 0.584 | 0.801 | 0.006 | 0.342 | 0.996       |
|                             |                                         | car   | 0.569 | 0.806 | 0.013 | 0.339 | 0.994       |
| MobileNet_SSD_Lite_v2.0     | ../predictor/_fixtures/lane_control.jpg | car   | 0.581 | 0.992 | 0.612 | 0.992 | 0.999       |
|                             |                                         | car   | 0.585 | 0.800 | 0.020 | 0.331 | 0.999       |
|                             |                                         | car   | 0.584 | 0.806 | 0.011 | 0.337 | 0.998       |
|                             |                                         | car   | 0.578 | 0.992 | 0.608 | 0.996 | 0.993       |
|                             |                                         | car   | 0.583 | 0.807 | 0.007 | 0.335 | 0.987       |

### Image Enhancement

| Name                        | Image                               | (R, G, B) at (0, 0) (top-left corner) |
|:---------------------------:|:-----------------------------------:|:-------------------------------------:|
| SRGAN_v1.0                  | ../predictor/_fixtures/penguin.png  | (0xc2, 0xc2, 0xc6)                    |
