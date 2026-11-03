import numpy as np

from mixes import song
from kuralkoodu.separation import sdr, separate


def test_builtin_separation_improves_sdr_and_sums_to_mix():
    mix, v, b = song(30, seed=3)
    voc, inst, info = separate(mix, 44100, "builtin")
    assert voc.shape == mix.shape and inst.shape == mix.shape
    assert np.allclose(voc + inst, mix, atol=1e-5)          # exact complement
    vm = voc.mean(axis=1)
    base = sdr(mix.mean(axis=1), v)                          # "do nothing" baseline
    got = sdr(vm, v)
    assert got - base > 10.0, (base, got)                    # measurable vocal improvement
    inst_base = sdr(mix, b)
    assert sdr(inst, b) - inst_base > 10.0


def test_mono_input_works():
    mix, v, _ = song(12, seed=4)
    voc, inst, info = separate(mix.mean(axis=1), 44100, "builtin")
    assert voc.shape == (len(mix), 1) and info["channels"] == 1
