from detectron2.config import CfgNode as CN


def add_extra_config(cfg):

    # extra configs for swin transformer
    cfg.MODEL.SWINT = CN()
    cfg.MODEL.SWINT.EMBED_DIM = 96
    cfg.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    cfg.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWINT.WINDOW_SIZE = 7
    cfg.MODEL.SWINT.MLP_RATIO = 4
    cfg.MODEL.SWINT.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWINT.APE = False
    cfg.MODEL.SWINT.VERSION = 1
    cfg.MODEL.SWINT.OUT_NORM = True
    cfg.MODEL.BACKBONE.FREEZE_AT = -1

    # extra configs for atss
    cfg.MODEL.ATSS = CN()
    cfg.MODEL.ATSS.TOPK = 9
    cfg.MODEL.ATSS.NUM_CLASSES = 80
    cfg.MODEL.ATSS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.ATSS.NUM_CONVS = 4
    cfg.MODEL.ATSS.CHANNELS = 256
    cfg.MODEL.ATSS.USE_GN = True

    cfg.MODEL.ATSS.IOU_THRESHOLDS = [0.4, 0.5]
    cfg.MODEL.ATSS.IOU_LABELS = [0, -1, 1]
    cfg.MODEL.ATSS.PRIOR_PROB = 0.01
    cfg.MODEL.ATSS.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
    cfg.MODEL.ATSS.FOCAL_LOSS_GAMMA = 2.0
    cfg.MODEL.ATSS.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.ATSS.REG_LOSS_WEIGHT = 2.0

    cfg.MODEL.ATSS.INFERENCE_TH = 0.05
    cfg.MODEL.ATSS.PRE_NMS_TOP_N = 1000
    cfg.MODEL.ATSS.NMS_TH = 0.6

    cfg.SOLVER.OPTIMIZER = 'SGD'