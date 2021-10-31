from detectron2.config import CfgNode as CN


def add_dyhead_config(cfg):
    """
    Add config for DYHEAD.
    """
    cfg.MODEL.DYHEAD = CN()
    # the number of convolutions used in the cls and bbox tower
    cfg.MODEL.DYHEAD.NUM_CONVS = 6
    # the channels of convolutions used in the cls and bbox tower
    cfg.MODEL.DYHEAD.CHANNELS = 256
