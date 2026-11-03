from networks.VNet import VNet, DuCiSC, MSPCNet


def net_factory(net_type="MSPCNet", in_chns=1, class_num=2, mode="train", use_sdm_head=True):
    has_dropout = mode == "train"
    if net_type == "vnet":
        net = VNet(
            n_channels=in_chns,
            n_classes=class_num,
            normalization="batchnorm",
            has_dropout=has_dropout,
        ).cuda()
    elif net_type in ("DuCiSC", "ducisc"):
        net = DuCiSC(
            n_channels=in_chns,
            n_classes=class_num,
            normalization="batchnorm",
            has_dropout=has_dropout,
        ).cuda()
    elif net_type in ("MSPCNet", "mspc", "MSPC"):
        net = MSPCNet(
            n_channels=in_chns,
            n_classes=class_num,
            normalization="batchnorm",
            has_dropout=has_dropout,
            use_sdm_head=use_sdm_head,
        ).cuda()
    else:
        raise ValueError(f"Unknown net_type: {net_type}")
    return net
