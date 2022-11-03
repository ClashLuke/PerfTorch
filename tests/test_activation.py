import threading
import time

import pytest
import torch

from perftorch.activation import Activation, GeLU, LeCunTanh, Mish, Swish


def mem_thread(mem_list):
    max_mem = 0
    while mem_list[0]:
        max_mem = max(max_mem, torch.cuda.memory_allocated())
    mem_list[1] = max_mem
    mem_list[0] = True


def core(instance: Activation, size: int):
    torch.cuda.empty_cache()
    inp = torch.randn((size,), device="cuda:0", dtype=torch.float64, requires_grad=True)
    out = instance(inp)
    out.mean().backward()
    del inp, out
    torch.cuda.empty_cache()


@pytest.mark.skip
def test_mem(instance: Activation, size: int) -> float:
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    mem_list = [True, 0]
    thread = threading.Thread(target=mem_thread, args=(mem_list,))
    thread.start()
    for i in range(64):
        core(instance, size)
    mem_list[0] = False
    thread.join()
    return mem_list[1]


@pytest.mark.parametrize("fn", [Mish, Swish, GeLU, LeCunTanh])
@pytest.mark.parametrize("size", [2 ** 24, 2 ** 26, 2 ** 28])
def test_activation_memory(fn: type, size: int):
    try:
        custom_mem = test_mem(fn(False), size)
        baseline_mem = test_mem(fn(True), size)
    except RuntimeError:
        torch.cuda.empty_cache()
        pytest.skip("OOM")
        return
    assert custom_mem < baseline_mem


def runtime(fn) -> float:
    for i in range(4):
        fn()
    start_time = time.time()
    for i in range(16):
        fn()
    return time.time() - start_time


@pytest.mark.parametrize("fn", [Mish, Swish, GeLU, LeCunTanh])
@pytest.mark.parametrize("size", [2 ** 24, 2 ** 26, 2 ** 28])
def test_activation_runtime(fn: type, size: int):
    torch.cuda.empty_cache()
    custom_time = runtime(lambda: core(fn(False), size))
    baseline_time = runtime(lambda: core(fn(True), size))

    assert custom_time < baseline_time
