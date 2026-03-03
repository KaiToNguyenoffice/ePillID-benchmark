"""
Image augmentation pipelines for pill images.

Supports two backends:
  - 'imgaug'         : original pipeline (legacy)
  - 'albumentations' : modern, faster alternative

Both backends expose the same API via ``get_augmentation_sequences()``.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend: imgaug (legacy)
# ---------------------------------------------------------------------------
def get_imgaug_sequences(low_gblur=1.0, high_gblur=3.0,
                         addgn_base_ref=0.01, addgn_base_cons=0.001,
                         rot_angle=180, max_scale=1.0,
                         add_perspective=False):
    from imgaug import augmenters as iaa

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    affine_seq = iaa.Sequential([
        iaa.Affine(
            rotate=(-rot_angle, rot_angle),
            scale=(0.8, max_scale),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        ),
        sometimes(iaa.Affine(shear=(-4, 4)))
    ])

    affine_list = [affine_seq]
    contrast_list = [
        iaa.Sequential([
            iaa.LinearContrast((0.7, 1.0), per_channel=False),
            iaa.Add((-30, 30), per_channel=False),
        ]),
        iaa.Sequential([
            iaa.LinearContrast((0.4, 1.0), per_channel=False),
            iaa.Add((-80, 80), per_channel=False),
        ])
    ]

    if add_perspective:
        logger.info("Adding perspective transform to augmentation")
        affine_list += [sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))]
        contrast_list += [
            iaa.GammaContrast((0.5, 1.7), per_channel=True),
            iaa.SigmoidContrast(gain=(8, 12), cutoff=(0.2, 0.8), per_channel=False)
        ]

    ref_seq = iaa.Sequential(affine_list + [
        iaa.OneOf(contrast_list),
        iaa.OneOf([
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 3 * addgn_base_ref * 255), per_channel=0.5),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, addgn_base_ref * 255), per_channel=0.5),
        ]),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0, high_gblur)),
            iaa.GaussianBlur(sigma=(0, low_gblur)),
        ])
    ])

    cons_seq = iaa.Sequential(affine_list + [
        iaa.LinearContrast((0.9, 1.1), per_channel=False),
        iaa.Add((-10, 10), per_channel=False),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 5 * addgn_base_cons * 255), per_channel=0.5),
        iaa.GaussianBlur(sigma=(0, low_gblur)),
    ])

    return affine_seq, ref_seq, cons_seq


# ---------------------------------------------------------------------------
# Backend: albumentations (modern)
# ---------------------------------------------------------------------------
def get_albumentations_sequences(low_gblur=1.0, high_gblur=3.0,
                                  addgn_base_ref=0.01, addgn_base_cons=0.001,
                                  rot_angle=180, max_scale=1.0,
                                  add_perspective=False):
    import albumentations as A

    common_spatial = [
        A.Affine(
            rotate=(-rot_angle, rot_angle),
            scale=(0.8, max_scale),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            shear=(-4, 4),
            p=1.0,
        ),
    ]

    if add_perspective:
        common_spatial.append(A.Perspective(scale=(0.01, 0.1), p=0.5))

    ref_color = [
        A.OneOf([
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(-30/255, 30/255),
                                           contrast_limit=(-0.3, 0.0), p=1.0),
            ]),
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(-80/255, 80/255),
                                           contrast_limit=(-0.6, 0.0), p=1.0),
            ]),
        ], p=1.0),
    ]

    if add_perspective:
        ref_color.append(A.RandomGamma(gamma_limit=(50, 170), p=0.3))

    ref_noise = [
        A.GaussNoise(std=(0, 3 * addgn_base_ref * 255), p=0.5),
        A.GaussianBlur(blur_limit=(3, max(3, int(high_gblur * 2) | 1)),
                       sigma_limit=(0, high_gblur), p=0.5),
    ]

    cons_color = [
        A.RandomBrightnessContrast(brightness_limit=(-10/255, 10/255),
                                   contrast_limit=(-0.1, 0.1), p=1.0),
    ]
    cons_noise = [
        A.GaussNoise(std=(0, 5 * addgn_base_cons * 255), p=0.5),
        A.GaussianBlur(blur_limit=(3, max(3, int(low_gblur * 2) | 1)),
                       sigma_limit=(0, low_gblur), p=0.5),
    ]

    ref_transform = A.Compose(common_spatial + ref_color + ref_noise)
    cons_transform = A.Compose(common_spatial + cons_color + cons_noise)

    class _AlbumentationsWrapper:
        """Makes albumentations pipeline compatible with the imgaug interface
        used by ``SingleImgPillID.load_img``."""
        def __init__(self, transform):
            self._transform = transform

        def augment_images(self, images):
            return [self._transform(image=img)['image'] for img in images]

    return (
        _AlbumentationsWrapper(A.Compose(common_spatial)),
        _AlbumentationsWrapper(ref_transform),
        _AlbumentationsWrapper(cons_transform),
    )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
def get_augmentation_sequences(backend='imgaug', **kwargs):
    """Return (affine_seq, ref_seq, cons_seq) using the selected backend.

    Parameters match ``get_imgaug_sequences``.
    """
    if backend == 'albumentations':
        logger.info("Using albumentations augmentation backend")
        return get_albumentations_sequences(**kwargs)
    else:
        logger.info("Using imgaug augmentation backend")
        return get_imgaug_sequences(**kwargs)
