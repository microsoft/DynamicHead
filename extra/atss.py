import logging
import math
from typing import Dict, List
import torch
from torch import Tensor, nn
import torch.distributed as dist

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat, batched_nms
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.comm import get_world_size
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from .sigmoid_focal_loss import SigmoidFocalLoss

__all__ = ["ATSS"]

logger = logging.getLogger(__name__)

INF = 1e8


def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []

    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)

    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


@META_ARCH_REGISTRY.register()
class ATSS(nn.Module):
    """
    Implement ATSS based on https://github.com/sfzhang15/ATSS
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        head: nn.Module,
        head_in_features,
        anchor_generator,
        box2box_transform,
        anchor_matcher,
        num_classes,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        box_reg_loss_weight=2.0,
        pre_nms_thresh=0.05,
        pre_nms_top_n=1000,
        nms_thresh=0.6,
        max_detections_per_image=100,
        pixel_mean,
        pixel_std,
        input_format="BGR",
        anchor_aspect_ratio,
        anchor_topk,
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.head_in_features = head_in_features
        if len(self.backbone.output_shape()) != len(self.head_in_features):
            logger.warning("[RetinaNet] Backbone produces unused features.")
        self.num_classes = num_classes
        self.box_coder = box2box_transform
        self.centerness_loss_func = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.classification_loss_func = SigmoidFocalLoss(focal_loss_gamma, focal_loss_alpha)

        # Anchors
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.anchor_aspect_ratio = anchor_aspect_ratio
        self.anchor_topk = anchor_topk

        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.box_reg_loss_weight = box_reg_loss_weight

        # Inference parameters:
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.max_detections_per_image = max_detections_per_image

        self.input_format = input_format
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.ATSS.IN_FEATURES]
        head = ATSSHead(cfg, feature_shapes)
        anchor_generator = build_anchor_generator(cfg, feature_shapes)
        return {
            "backbone": backbone,
            "head": head,
            "anchor_generator": anchor_generator,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ATSS.BBOX_REG_WEIGHTS),
            "anchor_matcher": Matcher(
                cfg.MODEL.ATSS.IOU_THRESHOLDS,
                cfg.MODEL.ATSS.IOU_LABELS,
                allow_low_quality_matches=True,
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.ATSS.NUM_CLASSES,
            "head_in_features": cfg.MODEL.ATSS.IN_FEATURES,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.ATSS.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.ATSS.FOCAL_LOSS_GAMMA,
            "box_reg_loss_weight": cfg.MODEL.ATSS.REG_LOSS_WEIGHT,
            # Inference parameters:
            "pre_nms_thresh": cfg.MODEL.ATSS.INFERENCE_TH,
            "pre_nms_top_n": cfg.MODEL.ATSS.PRE_NMS_TOP_N,
            "nms_thresh": cfg.MODEL.ATSS.NMS_TH,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "input_format": cfg.INPUT.FORMAT,
            "anchor_aspect_ratio": cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,
            "anchor_topk": cfg.MODEL.ATSS.TOPK
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]

        anchors = self.anchor_generator(features)
        pred_logits, pred_anchor_deltas, pred_centers = self.head(features)

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
            losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, pred_centers)

            return losses
        else:
            results = self.inference(anchors, pred_logits, pred_anchor_deltas, pred_centers, images.image_sizes)
            if torch.jit.is_scripting():
                return results
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, pred_centers):
        N = len(gt_labels)
        box_cls_flatten, box_regression_flatten = concat_box_prediction_layers(pred_logits, pred_anchor_deltas)
        centerness_flatten = [ct.permute(0, 2, 3, 1).reshape(N, -1, 1) for ct in pred_centers]
        centerness_flatten = torch.cat(centerness_flatten, dim=1).reshape(-1)

        labels_flatten = torch.cat(gt_labels, dim=0)
        reg_targets_flatten = torch.cat(gt_boxes, dim=0)
        anchors_flatten = torch.cat([anchors_per_image.tensor for anchors_per_image in anchors for _ in range(N)], dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)
        cls_loss = self.classification_loss_func(box_cls_flatten, labels_flatten.int()) / num_pos_avg_per_gpu

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        anchors_flatten = anchors_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten, anchors_flatten)

            sum_centerness_targets_avg_per_gpu = reduce_sum(centerness_targets.sum()).item() / float(num_gpus)
            reg_loss = self.GIoULoss(box_regression_flatten, reg_targets_flatten, anchors_flatten,
                                     weight=centerness_targets) / sum_centerness_targets_avg_per_gpu
            centerness_loss = self.centerness_loss_func(centerness_flatten, centerness_targets) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()

        return {
            "loss_cls": cls_loss,
            "loss_ctr": centerness_loss,
            "loss_box_reg": reg_loss*self.box_reg_loss_weight,
        }

    @torch.no_grad()
    def label_anchors(self, anchors, targets):
        cls_labels = []
        reg_targets = []
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i].gt_boxes
            bboxes_per_im = targets_per_im.tensor
            labels_per_im = targets[im_i].gt_classes + 1
            anchors_per_im = Boxes.cat(anchors)
            num_gt = bboxes_per_im.shape[0]

            num_anchors_per_loc = len(self.anchor_aspect_ratio)
            num_anchors_per_level = [len(anchors_per_level) for anchors_per_level in anchors]
            ious = pairwise_iou(anchors_per_im, targets_per_im)

            gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
            gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
            gt_points = torch.stack((gt_cx, gt_cy), dim=1)

            anchors_cx_per_im = (anchors_per_im.tensor[:, 2] + anchors_per_im.tensor[:, 0]) / 2.0
            anchors_cy_per_im = (anchors_per_im.tensor[:, 3] + anchors_per_im.tensor[:, 1]) / 2.0
            anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

            distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

            # Selecting candidates based on the center distance between anchor box and object
            candidate_idxs = []
            star_idx = 0
            for level, anchors_per_level in enumerate(anchors):
                end_idx = star_idx + num_anchors_per_level[level]
                distances_per_level = distances[star_idx:end_idx, :]
                topk = min(self.anchor_topk*num_anchors_per_loc, num_anchors_per_level[level])
                _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
                candidate_idxs.append(topk_idxs_per_level + star_idx)
                star_idx = end_idx
            candidate_idxs = torch.cat(candidate_idxs, dim=0)

            # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
            candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
            iou_mean_per_gt = candidate_ious.mean(0)
            iou_std_per_gt = candidate_ious.std(0)
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
            is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

            # Limiting the final positive samples’ center to object
            anchor_num = anchors_cx_per_im.shape[0]
            for ng in range(num_gt):
                candidate_idxs[:, ng] += ng * anchor_num
            e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            candidate_idxs = candidate_idxs.view(-1)
            l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 0]
            t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
            r = bboxes_per_im[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
            b = bboxes_per_im[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
            is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
            is_pos = is_pos & is_in_gts

            # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
            ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
            index = candidate_idxs.view(-1)[is_pos.view(-1)]
            ious_inf[index] = ious.t().contiguous().view(-1)[index]
            ious_inf = ious_inf.view(num_gt, -1).t()

            anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
            cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
            cls_labels_per_im[anchors_to_gt_values == -INF] = 0
            matched_gts = bboxes_per_im[anchors_to_gt_indexs]

            reg_targets_per_im = self.box_coder.get_deltas(anchors_per_im.tensor, matched_gts)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return cls_labels, reg_targets

    def compute_centerness_targets(self, reg_targets, anchors):
        gts = self.box_coder.apply_deltas(reg_targets, anchors)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l = anchors_cx - gts[:, 0]
        t = anchors_cy - gts[:, 1]
        r = gts[:, 2] - anchors_cx
        b = gts[:, 3] - anchors_cy
        left_right = torch.stack([l, r], dim=1)
        top_bottom = torch.stack([t, b], dim=1)
        centerness = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                                (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    def GIoULoss(self, pred, target, anchor, weight=None):
        pred_boxes = self.box_coder.apply_deltas(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        gt_boxes = self.box_coder.apply_deltas(target.view(-1, 4), anchor.view(-1, 4))
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()

    def inference(self, anchors, pred_logits, pred_anchor_deltas, pred_centers, image_sizes):
        """
        Arguments:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            pred_logits, pred_anchor_deltas: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
            image_sizes (List[(h, w)]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results: List[Instances] = []

        boxes_all, scores_all, labels_all = [], [], []
        for anchor_per_feat, pred_logits_per_feat, deltas_per_feat, centers_per_feat in \
                zip(anchors, pred_logits, pred_anchor_deltas, pred_centers):
            boxes, scores, labels = self.inference_single_feature_map(
                anchor_per_feat, pred_logits_per_feat, deltas_per_feat, centers_per_feat, image_sizes
            )
            boxes_all.append(boxes)
            scores_all.append(scores)
            labels_all.append(labels)

        boxes_all = list(zip(*boxes_all))
        scores_all = list(zip(*scores_all))
        labels_all = list(zip(*labels_all))

        for boxes_per_image, scores_per_image, labels_per_image, image_size \
                in zip(boxes_all, scores_all, labels_all, image_sizes):
            boxes_per_image, scores_per_image, labels_per_image = [
                cat(x) for x in [boxes_per_image, scores_per_image, labels_per_image]
            ]
            keep = batched_nms(boxes_per_image, scores_per_image, labels_per_image, self.nms_thresh)
            keep = keep[: self.max_detections_per_image]

            result = Instances(image_size)
            result.pred_boxes = Boxes(boxes_per_image[keep])
            result.scores = scores_per_image[keep]
            result.pred_classes = labels_per_image[keep]
            results.append(result)
        return results

    def inference_single_feature_map(self, anchors, box_cls, box_delta, centerness, image_sizes):
        N, _, H, W = box_cls.shape
        A = box_delta.size(1) // 4
        C = box_cls.size(1) // A

        # put in the same format as anchors
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()

        box_regression = permute_and_flatten(box_delta, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        centerness = permute_and_flatten(centerness, N, A, 1, H, W)
        centerness = centerness.reshape(N, -1).sigmoid()

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        boxes, scores, labels = [], [], []
        for per_box_cls, per_box_regression, per_pre_nms_top_n, per_candidate_inds, per_imsize \
                in zip(box_cls, box_regression, pre_nms_top_n, candidate_inds, image_sizes):

            per_box_cls = per_box_cls[per_candidate_inds]
            per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
            per_candidate_nonzeros = per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            detections = self.box_coder.apply_deltas(
                per_box_regression[per_box_loc, :].view(-1, 4),
                anchors.tensor[per_box_loc, :].view(-1, 4)
            )

            boxes.append(detections)
            scores.append(per_box_cls)
            labels.append(per_class)

        return boxes, scores, labels

    def preprocess_image(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class Scale(torch.nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = torch.nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ATSSHead(torch.nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super(ATSSHead, self).__init__()
        self.cfg = cfg
        num_classes = cfg.MODEL.ATSS.NUM_CLASSES
        num_anchors = len(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS)
        channels = cfg.MODEL.ATSS.CHANNELS
        use_gn = cfg.MODEL.ATSS.USE_GN

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        if cfg.MODEL.ATSS.NUM_CONVS>0:
            cls_tower = []
            bbox_tower = []
            for i in range(cfg.MODEL.ATSS.NUM_CONVS):
                cls_tower.append(
                    nn.Conv2d(
                        in_channels if i==0 else channels,
                        channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True
                    )
                )
                if use_gn:
                    cls_tower.append(nn.GroupNorm(32, channels))

                cls_tower.append(nn.ReLU())

                bbox_tower.append(
                    nn.Conv2d(
                        in_channels if i == 0 else channels,
                        channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True
                    )
                )
                if use_gn:
                    bbox_tower.append(nn.GroupNorm(32, channels))

                bbox_tower.append(nn.ReLU())

            self.add_module('cls_tower', nn.Sequential(*cls_tower))
            self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        else:
            self.cls_tower = None
            self.bbox_tower = None

        self.cls_logits = nn.Conv2d(channels, num_anchors * num_classes, kernel_size=1)
        self.bbox_pred = nn.Conv2d(channels, num_anchors * 4, kernel_size=1)
        self.centerness = nn.Conv2d(channels, num_anchors * 1, kernel_size=1)

        # initialization
        if cfg.MODEL.ATSS.NUM_CONVS>0:
            for modules in [self.cls_tower, self.bbox_tower]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)

        for modules in [self.cls_logits, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.ATSS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            if self.cls_tower is None:
                cls_tower = feature
            else:
                cls_tower = self.cls_tower(feature)

            if self.bbox_tower is None:
                box_tower = feature
            else:
                box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            bbox_reg.append(bbox_pred)

            centerness.append(self.centerness(box_tower))
        return logits, bbox_reg, centerness