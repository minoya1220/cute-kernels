import torch
import pynvml
import threading, time
from collections.abc import Iterable, Callable


def benchmark(fn: Callable, inputs: Iterable[torch.Tensor] | torch.Tensor = (), outputs: Iterable[torch.Tensor] | torch.Tensor = (),*, warmup=20, iters=100, L2_clear=True):
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)

    inputs_buffers = [inputs]
    outputs_buffers = [outputs]
    n_buffers = 1

    if L2_clear:
        assert inputs and outputs, "Input and Output Tensors are required to rotate L2"
        size = sum([t.nbytes for t in (*inputs, *outputs)])
        buffer_size = 256 * 2**20 # 256 MB clears l2 for all modern gpus
        n_buffers =  min(iters, buffer_size // size + 1)

        inputs_buffers = [tuple(t.clone() for t in inputs) for _ in range(0, n_buffers)]
        outputs_buffers = [tuple(t.clone() for t in outputs) for _ in range(0, n_buffers)]

    num_outs = len(outputs_buffers[0])
    if num_outs == 0:
        caller = lambda i, o: fn(*i)
    elif num_outs == 1:
        caller = lambda i, o: fn(*i, out=o[0]) 
    else:
        caller = lambda i, o: fn(*i, out=o)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(0, warmup):
            caller(inputs_buffers[i % n_buffers], outputs_buffers[i % n_buffers])
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(0, iters):
            caller(inputs_buffers[i % n_buffers], outputs_buffers[i % n_buffers])
    
    torch.cuda.synchronize()
    times = []
    for i in range(5):
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / iters)
    times.sort()
    median_time = times[len(times) // 2]
    return median_time


class MeasureMemoryClock():
    def __init__(self, interval=0.05):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.samples = []
        self._stop = threading.Event()
        self.interval = interval
        
    def sample_clk(self, interval):
        while not self._stop.is_set():
            self.samples.append(
                pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            )
            self._stop.wait(interval)
    
    def __enter__(self):
        self.thread = threading.Thread(target=self.sample_clk, args=(self.interval,), daemon=True)
        self.thread.start()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        self.thread.join()
        self.samples.sort()
        self.median = self.samples[len(self.samples) // 2]
