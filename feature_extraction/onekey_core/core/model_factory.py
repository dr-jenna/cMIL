# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2019/7/24
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2019 All Rights Reserved.

from onekey_core import models
__all__ = ['create_model']


def create_model(model_name, **kwargs):
    """Create core that torch vision supported. Supported `model_name` is as followings.
        ResNet, resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet50_2, wide_resnet101_2,
        resnext50_32x4d, resnext101_32x8d,
        detection.*,
        segmentation.*

    :param model_name: The above core name.
    :param kwargs: other core settings.
    :return: the matched core.
    :raise: ValueError, whose `model_name` is not supported.
    """
    supported_models = [k for k in models.__dict__
                        if not k.startswith('_') and type(models.__dict__[k]).__name__ != 'module']
    supported_models.extend(['detection', 'segmentation', 'segmentation3d', 'classification3d', 'fusion'])
    _modules = model_name.split('.')
    if len(_modules) == 1:
        if _modules[0] in supported_models:
            return models.__dict__[_modules[0]](**kwargs)
    elif len(_modules) == 2:
        if _modules[0] in supported_models and _modules[1] in models.__dict__[_modules[0]].__dict__:
            return models.__dict__[_modules[0]].__dict__[_modules[1]](**kwargs)

    raise ValueError(f'{model_name} not supported!')
