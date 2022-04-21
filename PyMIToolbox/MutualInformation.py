from ctypes import POINTER, c_uint, c_int, c_double

import numpy as np

from PyMIToolbox import _mitoolbox


def calcMutualInformation(feature, target):
    length = len(target)
    feature = np.ascontiguousarray(feature, dtype=np.uint32)
    target = np.ascontiguousarray(target, dtype=np.uint32)

    feature = feature.ctypes.data_as(POINTER(c_uint))
    target = target.ctypes.data_as(POINTER(c_uint))

    return _calcMutualInformation(feature, target, length)


def discAndCalcMutualInformation(feature, target):
    length = len(target)
    feature = np.ascontiguousarray(feature, dtype=np.double)
    target = np.ascontiguousarray(target, dtype=np.double)

    feature = feature.ctypes.data_as(POINTER(c_double))
    target = target.ctypes.data_as(POINTER(c_double))

    return _discAndCalcMutualInformation(feature, target, length)


def calcConditionalMutualInformation(feature_i, target, feature_j):
    length = len(target)
    feature_i = np.ascontiguousarray(feature_i, dtype=np.uint32)
    target = np.ascontiguousarray(target, dtype=np.uint32)
    feature_j = np.ascontiguousarray(feature_j, dtype=np.uint32)

    feature_i = feature_i.ctypes.data_as(POINTER(c_uint))
    target = target.ctypes.data_as(POINTER(c_uint))
    feature_j = feature_j.ctypes.data_as(POINTER(c_uint))

    return _calcConditionalMutualInformation(feature_i, target, feature_j, length)


def discAndCalcConditionalMutualInformation(feature_i, target, feature_j):
    length = len(target)
    feature_i = np.ascontiguousarray(feature_i, dtype=np.double)
    target = np.ascontiguousarray(target, dtype=np.double)
    feature_j = np.ascontiguousarray(feature_j, dtype=np.double)

    feature_i = feature_i.ctypes.data_as(POINTER(c_double))
    target = target.ctypes.data_as(POINTER(c_double))
    feature_j = feature_j.ctypes.data_as(POINTER(c_double))

    return _discAndCalcConditionalMutualInformation(feature_i, target, feature_j, length)


_calcMutualInformation = _mitoolbox.calcMutualInformation
_calcMutualInformation.argtypes = [POINTER(c_uint), POINTER(c_uint), c_int]
_calcMutualInformation.restype = c_double

_discAndCalcMutualInformation = _mitoolbox.discAndCalcMutualInformation
_discAndCalcMutualInformation.argtypes = [POINTER(c_double), POINTER(c_double), c_int]
_discAndCalcMutualInformation.restype = c_double

_calcConditionalMutualInformation = _mitoolbox.calcConditionalMutualInformation
_calcConditionalMutualInformation.argtypes = [POINTER(c_uint), POINTER(c_uint), POINTER(c_uint), c_int]
_calcConditionalMutualInformation.restype = c_double

_discAndCalcConditionalMutualInformation = _mitoolbox.discAndCalcConditionalMutualInformation
_discAndCalcConditionalMutualInformation.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int]
_discAndCalcConditionalMutualInformation.restype = c_double
