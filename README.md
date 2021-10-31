
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

~~Code and Model are under internal review and will release soon. Stay tuned!~~

In order to open-source, we have ported the implementation from our internal framework to Detectron2 and re-train the models.

We notice better performances on some models compared to original paper.

| Config                                                                       |           Model         |   Backbone  | Scheduler | COCO mAP |  Weight                                                                                |       
|------------------------------------------------------------------------------|-------------------------|-------------|-----------|----------|----------------------------------------------------------------------------------------|                                                                                                                                                      
|  [cfg](configs/dyhead_r50_rcnn_fpn_1x.yaml)                                  |    FasterRCNN + DyHead  |   R50       | 1x        | 40.3     |  [weight](https://xiyang.blob.core.windows.net/public/dyhead_r50_rcnn_fpn_1x.pth)      |       
|  [cfg](configs/dyhead_r50_retina_fpn_1x.yaml)                                |    RetinaNet + DyHead   |   R50       | 1x        | 39.9     |  [weight](https://xiyang.blob.core.windows.net/public/dyhead_r50_retina_fpn_1x.pth)    |            
|  [cfg](configs/dyhead_r50_atss_fpn_1x.yaml)                                  |    ATSS + DyHead        |   R50       | 1x        | 42.4     |  [weight](https://xiyang.blob.core.windows.net/public/dyhead_r50_atss_fpn_1x.pth)      |   
|  [cfg](configs/dyhead_swint_atss_fpn_2x_ms.yaml)                             |    ATSS + DyHead        |   Swin-Tiny | 2x + ms   | 49.8     |  [weight](https://xiyang.blob.core.windows.net/public/dyhead_swint_atss_fpn_2x_ms.pth) |    


### Usage
**Dependencies:**

[Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html), [timm](https://rwightman.github.io/pytorch-image-models/)

**Installation:**

```
python -m pip install -e DynamicHead
```

**Train:**

To train a config on a single node with 8 gpus, simply use:
```
DETECTRON2_DATASETS=$DATASET python train_net.py --config configs/dyhead_r50_retina_fpn_1x.yaml --num-gpus 8
```

**Test:**

To test a config with a weight on a single node with 8 gpus, simply use:
```
DETECTRON2_DATASETS=$DATASET python train_net.py --config configs/dyhead_r50_retina_fpn_1x.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS $WEIGHT
```


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

