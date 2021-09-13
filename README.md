
<div align="center">   

# Dynamic Head: Unifying Object Detection Heads with Attentions
</div>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dynamic-head-unifying-object-detection-heads/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=dynamic-head-unifying-object-detection-heads)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dynamic-head-unifying-object-detection-heads/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=dynamic-head-unifying-object-detection-heads)

https://user-images.githubusercontent.com/1438231/122347136-9282e900-cefe-11eb-8b36-ebe08736ec97.mp4


This is the official implementation of CVPR 2021 paper "Dynamic Head: Unifying Object Detection Heads with Attentions". 

_"In this paper, we present a novel dynamic head framework to unify object detection heads with attentions. 
By coherently combining multiple self-attention mechanisms between feature levels for scale-awareness, among spatial locations for spatial-awareness, and within output channels for task-awareness, the proposed approach significantly improves the representation ability of object detection heads without any computational overhead."_


>[**Dynamic Head: Unifying Object Detection Heads With Attentions**](https://arxiv.org/pdf/2106.08322.pdf)
>
>[Xiyang Dai](https://scholar.google.com/citations?user=QC8RwcoAAAAJ&hl=en), [Yinpeng Chen](https://scholar.google.com/citations?user=V_VpLksAAAAJ&hl=en), [Bin Xiao](https://scholar.google.com/citations?user=t5HZdzoAAAAJ&hl=en), [Dongdong Chen](https://scholar.google.com/citations?user=sYKpKqEAAAAJ&hl=zh-CN), [Mengchen Liu](https://scholar.google.com/citations?user=cOPQtYgAAAAJ&hl=zh-CN), [Lu Yuan](https://scholar.google.com/citations?user=k9TsUVsAAAAJ&hl=en), [Lei Zhang](https://scholar.google.com/citations?user=fIlGZToAAAAJ&hl=en) 



### Model Zoo

Code and Model are under internal review and will release soon. Stay tuned!


### Citation

```BibTeX
@InProceedings{Dai_2021_CVPR,
    author    = {Dai, Xiyang and Chen, Yinpeng and Xiao, Bin and Chen, Dongdong and Liu, Mengchen and Yuan, Lu and Zhang, Lei},
    title     = {Dynamic Head: Unifying Object Detection Heads With Attentions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {7373-7382}
}
```



### Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

------
# My Notes

Hi there, I am a recent undergrad graduate and am currently looking for ML positions. I always wanted to learn how to implement code from a paper, and I was happy to implement the DyHead attachment that can be used by others. 

All the code I wrote uses PyTorch, these are all the modules:
1. [`concat_fpn_output.py`](./torch/concat_fpn_output.py) - This takes the output of the FPN and concatenates all the levels to the median height and width of all the levels via upsampling or downsampling.
2. [`attention_layers.py`](./torch/attention_layers.py) - This contains all the classes for the three attention mechanisms.
 - Big Thanks to user Github [Islanna](https://github.com/Islanna/), she implemented code from the [Dynamic ReLU Paper](https://arxiv.org/pdf/2003.10027.pdf). The Task-aware Attention layer uses the same technique from Dynamic-ReLU-A that constructs a dynamic ReLU funtion that are both spatial and channel shared. I used her code as a way to understand how to implement it and I used the same techniques but made the code simpler for my own learning process, but all credits to her. This is her repository: https://github.com/Islanna/DynamicReLU.

4. [`DyHead.py`](./torch/DyHead.py) - This contains the classes to construct a single DyHead block or the entire DyHead.

The [`DyHead_Example.ipynb`](./torch/DyHead_Example.ipynb) notebook demonstrates how all the classes above work, I would encourage to have a look.

The code used is not the most efficient, but the code is well documented and easily understandable. However, I am sure changes to make it more efficient is not a problem.

## Future Additions:
The code does not contruct a full Object Detection model with a DyHead. This is the case because I currently need to change my focus on to just find a new position but also I was confused about the implementation of ROI Pooling on the tensor *F* since dimensions do not contain the spacial dimensions since it was reshaped to be LxSxC not LxHxWxC. I would like to hear more about how this is implemented.

So in the future when I have more time and a better understanding, I would like to implement both one-stage and two-stage detectors using PyTorch's Built-in FasterRCNN modules to easily adapt the inclusion of DyHead for detection purposes.
