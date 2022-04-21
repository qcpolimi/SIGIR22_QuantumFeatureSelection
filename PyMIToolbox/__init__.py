from ctypes import CDLL


def get_libpath():
    from platform import system
    from os.path import dirname, abspath, realpath, join

    root = dirname(abspath(realpath(__file__)))

    libdir = f'{root}/MIToolbox'
    libname = 'libMIToolbox'

    sys_name = system()
    if sys_name in ('Linux', 'Darwin'):
        libname += '.so'
    elif sys_name == 'Windows':
        libname += '.dll'
    else:
        raise RuntimeError("unsupported platform - \"{}\"".format(sys_name))

    return f'{libdir}/{libname}'


_mitoolbox = CDLL(get_libpath())

from .MutualInformation import calcMutualInformation, discAndCalcMutualInformation, calcConditionalMutualInformation, \
    discAndCalcConditionalMutualInformation
