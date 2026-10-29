"""Estimate return times of a time-averaged OU observable with GKTL.

This reproduces the benchmark of Ragone et al. 2018 (PNAS, SI Fig. S2). GKTL biases
the ensemble with exp(k * int x dt), and rare events are defined on the time-averaged
observable:

    A(T, t) = 1/T int_t^{t+T} x(s) ds

For each (backward reconstructed) trajectory n of duration T_a, the maximum
a_n = max_{0 <= t <= T_a - T} A(T, t) is computed together with its probability
p_n = exp(-k int_0^{T_a} x dt) prod_i R_i / N. The return time is then (Eq. S13):

    r(a) = -(T_a - T) / ln(1 - sum_n p_n 1[a_n > a])

The estimate is compared with the return time sampled from a long control run
of duration T_d with the same Euler-Maruyama scheme, using block maxima of
duration T_a - T (Eq. S12).
"""

import argparse
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import toml
from ornsteinuhlenbeck import OrnsteinUhlenbeck
from scipy.signal import lfilter
from pyrevs.sampler import build_sampler


def running_mean_max(times: npt.NDArray[np.number], x: npt.NDArray[np.number], window: float) -> float:
    """Return the maximum of the running mean of x over a time window.

    Args:
        times: the times, uniformly spaced
        x: the values of x at times
        window: the averaging window T

    Returns:
        max_{0 <= t <= t_end - T} 1/T int_t^{t+T} x(s) ds
    """
    dt = times[1] - times[0]
    m = round(window / dt)
    cumint = np.concatenate([[0.0], np.cumsum(0.5 * (x[1:] + x[:-1]) * dt)])
    return float(np.max(cumint[m:] - cumint[:-m]) / window)


def control_run_return_time(
    duration: float, dt: float, window: float, block: float, theta: float, epsilon: float
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.number]]:
    """Sample the return time of A(T, t) from a long run, using block maxima (Eq. S12).

    The same Euler-Maruyama scheme as the OrnsteinUhlenbeck model is used. The run
    is generated in chunks to limit memory usage.

    Returns:
        the ranked block maxima and their return times
    """
    rng = np.random.default_rng()
    rho = 1.0 - theta * dt
    m = round(window / dt)
    nb = round(block / dt)
    nchunk = max(10 * nb, 10_000_000 // nb * nb)
    nsteps = round(duration / dt)

    x_last = rng.normal(scale=np.sqrt(epsilon / theta))
    x_tail = np.array([x_last])  # last m+1 values of x, to continue the running mean
    pending = np.array([])  # running mean values not yet assigned to a full block
    maxima = []
    for start in range(0, nsteps, nchunk):
        noise = np.sqrt(2.0 * epsilon * dt) * rng.standard_normal(min(nchunk, nsteps - start))
        # AR(1) recursion x_{i+1} = (1 - theta dt) x_i + noise_{i+1}
        x, _ = lfilter([1.0], [1.0, -rho], noise, zi=[rho * x_last])
        x_last = x[-1]
        xx = np.concatenate([x_tail, x])
        cumint = np.concatenate([[0.0], np.cumsum(0.5 * (xx[1:] + xx[:-1]) * dt)])
        if len(xx) > m:
            pending = np.concatenate([pending, (cumint[m:] - cumint[:-m]) / window])
        x_tail = xx[-(m + 1) :]
        nfull = len(pending) // nb
        maxima.append(pending[: nfull * nb].reshape(nfull, nb).max(axis=1))
        pending = pending[nfull * nb :]

    ranked = np.sort(np.concatenate(maxima))[::-1]
    nblocks = len(ranked)
    ranks = np.arange(1, nblocks)
    return ranked[:-1], -block / np.log(1.0 - ranks / nblocks)


def run_gktl(params: dict, nruns: int, window: float, out_file: Path) -> tuple[list, list]:
    """Run GKTL nruns times, returning the maxima a_n and probabilities p_n of each run."""
    maxima, probas = [], []
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.toml"
        with input_file.open("w") as f:
            toml.dump(params, f)
        for i in range(nruns):
            sampler = build_sampler(fmodel_t=OrnsteinUhlenbeck, a_args=["-i", str(input_file)])
            sampler.run()
            trajs = sampler.database.traj_list()
            maxima.append([running_mean_max(t.get_time_array(), t.get_score_array(), window) for t in trajs])
            probas.append(sampler.database.extension().clone_weights())
            del sampler
            np.savez(out_file, maxima=np.array(maxima), probas=np.array(probas))
            print(f"[{i}] sum p_n = {np.sum(probas[-1]):.3f}, max a_n = {np.max(maxima[-1]):.3f}")  # noqa: T201
    return maxima, probas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", default="input_gktl.toml", help="GKTL input file")
    parser.add_argument("-K", "--nruns", type=int, default=10, help="Number of independent GKTL runs")
    parser.add_argument("-T", "--window", type=float, default=50.0, help="Time averaging window")
    parser.add_argument("-o", "--output", default="gktl_ou_runs.npz", help="File storing the GKTL results")
    parser.add_argument("--from-file", action="store_true", help="Load the GKTL results instead of running")
    parser.add_argument("--control", type=float, default=1e9, help="Duration of the control run (0 to skip)")
    parser.add_argument("--plot", action="store_true", help="Plot the return time curves")
    args = parser.parse_args()

    with Path(args.input).open("r") as f:
        params = toml.load(f)
    theta = params["model"].get("theta", 1.0)
    epsilon = params["model"].get("epsilon", 0.5)
    t_a = params["gktl"]["end_time"]
    block = t_a - args.window

    out_file = Path(args.output)
    if args.from_file:
        data = np.load(out_file)
        maxima, probas = data["maxima"], data["probas"]
    else:
        maxima, probas = run_gktl(params, args.nruns, args.window, out_file)
    maxima, probas = np.array(maxima), np.array(probas)
    nruns = len(maxima)

    # Exceedance probability of each run (unbiased), averaged over the runs
    a_range = np.linspace(0.2, 0.9, 29)
    q = np.array([[np.sum(p[m > a]) for a in a_range] for m, p in zip(maxima, probas, strict=True)])
    q_mean = q.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_gktl = -block / np.log1p(-q_mean)
        r_runs = -block / np.log1p(-q)

    if args.control > 0.0:
        a_ctrl, r_ctrl = control_run_return_time(
            args.control, params["trajectory"]["step_size"], args.window, block, theta, epsilon
        )
        # Control run return time at the thresholds, where it was sampled
        r_ref = np.interp(a_range, a_ctrl[::-1], r_ctrl[::-1], left=np.nan, right=np.nan)
    else:
        r_ref = np.full_like(a_range, np.nan)

    print(f"\nGKTL ({nruns} runs, k = {params['gktl']['k']}), T = {args.window}, T_a = {t_a}")  # noqa: T201
    print(f"{'a':>6} {'q(a)':>11} {'r_GKTL':>11} {'r_control':>11} {'ratio':>7}")  # noqa: T201
    for j, a in enumerate(a_range):
        if q_mean[j] > 0.0:
            ratio = r_gktl[j] / r_ref[j]
            print(f"{a:6.3f} {q_mean[j]:11.3e} {r_gktl[j]:11.3e} {r_ref[j]:11.3e} {ratio:7.3f}")  # noqa: T201

    if args.plot:
        plt.figure(figsize=(6, 4))
        if args.control > 0.0:
            plt.plot(r_ctrl, a_ctrl, color="k", linestyle="--", linewidth=0.8, label=f"Control run ($T_d$ = {args.control:.0e})")
        for r in r_runs:
            plt.plot(r, a_range, color="tab:blue", linewidth=0.5, alpha=0.3)
        plt.plot(r_gktl, a_range, color="tab:blue", linewidth=1.5, label=f"GKTL ({nruns} runs)")
        plt.xscale("log")
        plt.xlim(left=1e2, right=1e10)
        plt.ylim(bottom=0.2, top=0.9)
        plt.xlabel(r"$r(a)$", fontsize="x-large")
        plt.ylabel(r"$a$", fontsize="x-large")
        plt.legend(fontsize="large")
        plt.grid(linestyle="dotted", color="silver")
        plt.tight_layout()
        plt.savefig("OU_ReturnTime_GKTL.png", dpi=300)
