import sys

assert 'dowel' not in sys.modules, 'dowel must be imported after dowel_wrapper.'

import dowel
dowel_eval = dowel
del sys.modules['dowel']

import dowel
dowel_plot = dowel
del sys.modules['dowel']

import dowel
dowel_tcp = dowel
del sys.modules['dowel']

import dowel
all_dowels = [dowel, dowel_eval, dowel_plot, dowel_tcp]
assert len(set(id(d) for d in all_dowels)) == len(all_dowels)

import aim_wrapper
def get_dowel(phase=None):
    if (phase or aim_wrapper.get_context().get('phase')).lower() == 'plot':
        return dowel_plot
    if (phase or aim_wrapper.get_context().get('phase')).lower() == 'eval':
        return dowel_eval
    if (phase or aim_wrapper.get_context().get('phase')).lower() == 'tcp':
        return dowel_tcp
    return dowel
def get_logger(phase=None):
    return get_dowel(phase).logger
def get_tabular(phase=None):
    return get_dowel(phase).tabular
