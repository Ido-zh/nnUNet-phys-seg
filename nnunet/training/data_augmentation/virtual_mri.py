import torch
import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform


def bSSFP_readout(M0, T1, T2, FA=35):
    """
    bSSFP imaging

    :param M0: Initial magnetization.
    :param T1: T1 map.
    :param T2: T2 map.
    :param FA: Flip angle in DEGREES.
    :return: Image readout.
    """
    cos_fa = np.cos(np.deg2rad(FA))
    sig = M0 / (1 + cos_fa + (1 - cos_fa) * T1 / np.clip(T2, 1e-3, None))
    return sig


def spin_echo_readout(M0, T1, T2, TR=1e3, TE=15):
    """
    Spin echo imaging.

    :param M0: Initial magnetization.
    :param T1: T1 map.
    :param T2: T2 map.
    :param TR: Repetition time.
    :param TE: Echo time
    :return: Image readout.
    """
    E1 = np.exp(-TR / np.clip(T1, 1e-3, None))
    E2 = np.exp(-TE / np.clip(T2, 1e-3, None))
    sig = M0 * (1. - E1) * E2
    return sig


def gradient_echo_readout(M0, T1, T2, FA=15, TR=50, TE=3):
    """
    Gradient echo imaging.

    :param M0: Initial magnetization.
    :param T1: T1 map.
    :param T2: T2 map.
    :param FA: Flip angle in DEGREES!
    :param TR: Repetition time.
    :param TE: Echo time
    :return: Image readout.
    """
    cos_fa = np.cos(np.deg2rad(FA))
    E1 = np.exp(-TR / np.clip(T1, 1e-3, None))
    E2 = np.exp(-TE / np.clip(T2, 1e-3, None))
    sig = M0 * (1. - E1) / (1 - cos_fa * E1) * E2
    return sig


def extract_blood_map(ssfp, seg):
    """
    Extract fuzzy blood map from SSFP cine image and its segmentation. Only support 2D operations!

    :param ssfp: (H, W)
    :param seg: (H, W)
    :return: left ventricle map, right ventricle map, blood map
    """
    # ensure ssfp has a minimum of 0
    ssfp = ssfp - ssfp.min()
    right_ventricle = seg == 1
    left_ventricle = seg > 1
    heart = left_ventricle | right_ventricle
    heart_signal = ssfp[heart]

    # intensity based segmentation: the brighter the signal is, the more likely it is blood.
    blood = np.zeros_like(ssfp)
    q10, q99 = np.percentile(heart_signal, (10, 99))
    heart_signal = (heart_signal - q10) / (q99 - q10)
    heart_signal = np.clip(heart_signal, 0., 1.)
    blood[heart] = heart_signal
    left_ventricle_blood = left_ventricle * blood
    right_ventricle_blood = right_ventricle * blood
    return left_ventricle_blood, right_ventricle_blood, blood


def blood_flow_decay(ssfp, seg, decay_duration: float = 3.):
    """
    Heuristic black blood imaging preparation. It's a bit hard to explain how it works.
    Only support 2D operations!

    :param ssfp: (H, W)
    :param seg: (H, W)
    :param decay_duration: should be in range (1, 20).
    :return: Decay factor of shape (H, W).
    """

    # ensure ssfp has a minimum of 0
    ssfp = ssfp - ssfp.min()
    heart = seg > 0
    heart_signal = ssfp[heart]

    decay_const = (ssfp - heart_signal.min()) * heart  # myocardium, background -> 0, blood -> large value
    decay_const = decay_const.max() - decay_const  # blood -> 0, the rest -> large value
    decay_max = np.percentile(decay_const[heart], 90)  # normalize
    decay_const = np.clip(decay_const / decay_max, 0., 1.) + 1
    decay = np.exp(-decay_duration / decay_const)  # signal decay factor
    decay = (decay - decay.min()) / (decay.max() - decay.min())  # normalize
    return decay


def triple_inversion_recovery_black_blood(ssfp, seg, M0, T1, T2,
                                          heuristic=True, STIR=False,
                                          TIb=0.7, TIf=0.7, T1fat=600, T2fat=150):
    """
    Triple inversion recovery for black-blood imaging with fat suppression.

    :param ssfp: (H, W)
    :param seg: (H, W)
    :param M0: Initial PD.
    :param T1: T1 map.
    :param T2: T2 map.
    :param heuristic: Use heuristic or real double inversion.
    :param STIR: Fat suppression option.
    :param TIb: Inversion time in Blood T1 unit, default is 0.7 (approx. ln(2)).
    :param TIf: Inversion time in Fat T1 unit, default is 0.7 (approx. ln(2)).
    :param T1fat: Fat T1 threshold for fat segmentation.
    :param T2fat: Fat T2 threshold for fat segmentation.
    :return: BB-prepared M0.
    """
    if heuristic:
        M0_decay = blood_flow_decay(ssfp, seg)
        M_prepared = M0 * M0_decay
    else:
        _, _, blood = extract_blood_map(ssfp, seg)
        T1blood = np.median(T1[blood > 0.5])
        M_flow = M0 * (1 - 2 * np.exp(-TIb * T1blood / T1))
        M_steady = M0
        M_DIR = np.abs(M_steady * (1. - blood) + M_flow * blood)
        M_prepared = M_DIR

    if STIR:
        # perform a third IR for fat suppression
        fat = (T1 < T1fat) & (T2 > T2fat)
        T1fat = np.median(T1[fat])
        M_prepared = np.abs(M_prepared * (1 - 2 * np.exp(-TIf * T1fat / T1)))

    return M_prepared


def gadolinium_flow(T1, flow_map, concentration_factor: float = 0.5, r1=4.5):
    """
    Simulate T1 variation with Gd. in blood.

    :param T1: Native T1 map.
    :param flow_map: Relative concentration map of Gd.
    :param concentration_factor: Concentration scale.
    :param r1: Relaxation rate of Gd.
    :return:
    """
    T1 = np.clip(T1, 1, None)
    concentration = concentration_factor * flow_map
    T1new = 1 / (concentration * r1 + 1e3 / T1) * 1e3
    return T1new


def saturation_recovery(M0, T1, TI: float = 100):
    """
    Saturation recovery preparation.

    :param M0: Magnetization before SR.
    :param T1: T1 map.
    :param TI: Inversion time.
    :return: SR: Magnetization after SR.
    """
    SR = M0 * (1. - np.exp(-TI / T1))
    return SR
