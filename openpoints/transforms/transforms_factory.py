import torch
from ..utils.registry import Registry

DataTransforms = Registry('datatransforms')


def concat_collate_fn(datas):
    """collate fn for point transformer
    """
    pts, labels, offset, count, batches = [], [], [], 0, []
    extras = {}
    for i, data in enumerate(datas):
        count += len(data['pos'])
        offset.append(count)
        pts.append(data['pos'])
        labels.append(data['y'])
        batches += [i] * len(data['pos'])
        for key, value in data.items():
            if key in {'pos', 'y', 'o', 'batch'}:
                continue
            extras.setdefault(key, []).append(value)
    collated = {
        'pos': torch.cat(pts),
        'y': torch.cat(labels),
        'o': torch.IntTensor(offset),
        'batch': torch.LongTensor(batches),
    }
    for key, values in extras.items():
        collated[key] = torch.cat(values)
    return collated


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, args):
        for t in self.transforms:
            args = t(args)
        return args


class ListCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coord, feat, label):
        for t in self.transforms:
            coord, feat, label = t(coord, feat, label)
        return coord, feat, label


def build_transforms_from_cfg(split, datatransforms_cfg):
    """
    Build a dataset transform for a certrain split, defined by `datatransforms_cfg`.
    """
    transform_list = datatransforms_cfg.get(split, None)
    transform_args = datatransforms_cfg.get('kwargs', None)
    compose_fn = eval(datatransforms_cfg.get('compose_fn', 'Compose'))
    if transform_list is None or len(transform_list) == 0:
        return None
    point_transforms = []
    if len(transform_list) > 1:
        for t in transform_list:
            point_transforms.append(DataTransforms.build(
                {'NAME': t}, default_args=transform_args))
        return compose_fn(point_transforms)
    else:
        return DataTransforms.build({'NAME': transform_list[0]}, default_args=transform_args)
