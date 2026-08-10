import torch
from softmax.softmax_fwd import softmax_fwd_builder
from softmax.softmax_bwd import softmax_bwd_builder
import matplotlib.pyplot as plt
from measure import benchmark
from matplotlib.ticker import FuncFormatter

BENCH_SIZE = 2**29

def test_fwd(dims: list[int], dtype=torch.bfloat16):
    for dim in dims:
        X = torch.randn(2**6 - 1, dim, device='cuda', dtype=dtype)

        torch_softmax = torch.softmax(X.double(), dim=1).to(X.dtype)
        softmax_fwd = softmax_fwd_builder(X)
        try:
            torch.testing.assert_close(torch_softmax, softmax_fwd(X))
            print(f"dim={dim} passed")

        except AssertionError as e:
            print(f"dim={dim} failed")

def test_bwd(dims: list[int], dtype=torch.bfloat16):
    for dim in dims:
        X = torch.randn(2**6 - 1, dim, device='cuda', dtype=dtype)
        dY = torch.randn_like(X)
        
        Y = torch.softmax(X, dim=1).to(dtype)
        Y_double, dY_double = Y.double(), dY.double()
        ref_softmax_bwd = (Y_double * (dY_double - (Y_double * dY_double).sum(dim=1, keepdim=True))).to(dtype)

        softmax_bwd = softmax_bwd_builder(Y, dY)
        try:
            torch.testing.assert_close(ref_softmax_bwd, softmax_bwd(Y, dY))
            print(f"dim={dim} passed")

        except AssertionError as e:
            print(f"dim={dim} failed")


def benchmark_fwd(dims: list[int], *, dtype=torch.bfloat16, elems=BENCH_SIZE) -> list[float]:
    times = []
    
    for dim in dims:
        X = torch.randn(elems // dim, dim, device='cuda', dtype=dtype)
        Y = torch.empty_like(X)
        softmax_fwd = softmax_fwd_builder(X)

        times.append(benchmark(softmax_fwd, [X], [Y]))
    
    return times


def benchmark_bwd(dims: list[int], *, dtype=torch.bfloat16, elems=BENCH_SIZE) -> list[float]:
    times = []
    for dim in dims:
        Y = torch.randn(elems//dim, dim, device='cuda', dtype=dtype)
        dY = torch.randn_like(Y)
        dX = torch.empty_like(Y)
        softmax_bwd = softmax_bwd_builder(Y, dY)
        times.append(benchmark(softmax_bwd, [Y, dY], [dX]))
    
    return times



def plot_bench(dims, times, passes, title, dtype=torch.bfloat16, *,
            elems=BENCH_SIZE, reference=False, memory_clock=None,
            color='#83a598', sizes=None, font='Lora', path=None):

    c = dict(bg='#1d2021', fg='#ebdbb2', muted='#928374', grid='#32302f',
            spine='#504945', band='#282828', line=color)
    s = {**dict(base=11.5, title=13.5, axis_label=11.5, tick=9.5, ref=9.5),
        **(sizes or {})}

    p = torch.cuda.get_device_properties(0)
    memory_clock = p.memory_clock_rate * 1e-3 if memory_clock is None else memory_clock
    peak = p.memory_bus_width / 8 * memory_clock * 1e6 * 2 / 1e9

    path = path or f"{title.lower().replace(' ', '_')}.png"

    esize = torch.finfo(dtype).bits // 8
    gb = [passes * elems * esize / (t * 1e-3) / 1e9 for t in times]

    lines = [(peak, ':', f'theoretical peak · {peak:.0f} GB/s')]

    if reference:
        ops = {
            2: (1, lambda a, *, out: out.copy_(a), 'torch copy (1r+1w)'),
            3: (2, torch.add, 'torch add (2r+1w)'),
        }
        if passes not in ops:
            raise ValueError(f"no reference op for passes={passes}, have {sorted(ops)}")
        n_in, fn, ref_label = ops[passes]

        ins = [torch.randn(elems, device='cuda', dtype=dtype) for _ in range(n_in)]
        out = torch.empty_like(ins[0])
        t_ref = benchmark(fn, ins, [out])
        ref = passes * ins[0].nbytes / (t_ref * 1e-3) / 1e9

        lines.append((ref, '--', f'{ref_label} · {ref:.0f} GB/s'))

    rc = {
        'font.family': 'serif',
        'font.serif': [font, 'TeX Gyre Pagella', 'DejaVu Serif'],
        'mathtext.fontset': 'cm', 'font.size': s['base'],
        'figure.facecolor': c['bg'], 'axes.facecolor': c['bg'],
        'savefig.facecolor': c['bg'],
        'text.color': c['fg'], 'axes.labelcolor': c['fg'],
        'xtick.color': c['muted'], 'ytick.color': c['muted'],
        'axes.edgecolor': c['spine'], 'axes.linewidth': 0.8,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'axes.spines.top': False, 'axes.spines.right': False,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(7.6, 4.8),
                            constrained_layout=dict(w_pad=0.18, h_pad=0.16))
        top, xr = peak * 1.10, dims[-1] * 1.42

        ax.set_axisbelow(True)
        ax.grid(axis='y', color=c['grid'], lw=0.7)
        ax.axhspan(peak, top, color=c['band'], lw=0, zorder=0)

        for y, ls, txt in lines:
            ax.axhline(y, ls=ls, lw=1.0, color=c['muted'], zorder=2)
            ax.annotate(txt, xy=(0.015, y), xycoords=ax.get_yaxis_transform(),
                        xytext=(0, 3), textcoords='offset points',
                        fontsize=s['ref'], color=c['muted'],
                        ha='left', va='bottom', zorder=4)

        ax.fill_between(dims, gb, color=c['line'], alpha=0.10, zorder=1)
        ax.plot(dims, gb, 'o-', ms=5, lw=2, color=c['line'], zorder=3)

        ax.set_xscale('log', base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims], rotation=45, ha='right',
                        rotation_mode='anchor', fontsize=s['tick'])
        ax.tick_params(axis='y', labelsize=s['tick'])
        ax.set_xlim(dims[0] / 1.55, xr)
        ax.set_ylim(0, top)
        ax.set_xlabel('row length $N$', labelpad=8, fontsize=s['axis_label'])
        ax.set_ylabel('achieved bandwidth (GB/s)', labelpad=8, fontsize=s['axis_label'])
        ax.set_title(title, fontsize=s['title'], loc='left', color=c['line'], pad=12)

        sec = ax.secondary_yaxis(
            'right', functions=(lambda g: g / peak * 100, lambda q: q / 100 * peak))
        sec.set_ylabel('% of theoretical peak', labelpad=10, fontsize=s['axis_label'])
        sec.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
        sec.spines['right'].set_visible(True)
        sec.spines['right'].set_color(c['spine'])
        sec.tick_params(colors=c['muted'], labelsize=s['tick'])
        sec.yaxis.label.set_color(c['fg'])

        fig.savefig(path, dpi=150)
        plt.close(fig)



if __name__ == "__main__":
    
    torch.manual_seed(123)

    test_dims = [3, 7, 8, 64, 100, 257, 1021, 2048, 2049, 4099, 8192, 32768, 131072]
    bench_dims = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    test_fwd(bench_dims + test_dims)
    test_bwd(bench_dims + test_dims)

    # with MeasureMemoryClock(interval=0.01) as mem_clk:
    # fwd_times = benchmark_fwd(bench_dims)
    # bwd_times = benchmark_bwd(bench_dims)
    
    # plot_bench(bench_dims, fwd_times, 2, 'Softmax Forward', color='#83a598', path='fwd_bench.png', reference=True)
    # plot_bench(bench_dims, bwd_times, 3, 'Softmax Backward', color='#fabd2f', path='bwd_bench.png', reference=True)
    
    # from torch.profiler import profile, ProfilerActivity
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    # ) as prof:
    #     benchmark_fwd([2**14])
    # prof.export_chrome_trace("trace.json")

    # test_fwd([2**14])

    # TODO: use pytest (maybe), rename repo, remove softmax pkg, fix plotting output dir, pack bf16s, do writeup, add controls to enable unoptimized versions
    
    print("success")
