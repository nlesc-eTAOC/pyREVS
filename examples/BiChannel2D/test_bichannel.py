import contextlib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import toml
from bichannel2d import BiChannel2D
from pyrevs.core import Config
from pyrevs.strategies.base import TargetScoreConvergenceCriterion
from pyrevs.trajectory import Trajectory
from pyrevs.trajectory import TrajectoryConfig

if __name__ == "__main__":
    # For convenience
    fmodel = BiChannel2D

    # Load the input file
    with Path("input.toml").open("r") as f:
        input_params = toml.load(f)

    # Setup parameters
    cfg = Config(input_params)
    tcfg = cfg.load(TrajectoryConfig)
    model_params = cfg.section_dict("model")

    # Initialize a trajectory object
    traj = Trajectory(traj_id=0, weight=1.0, fmodel_t=fmodel, traj_cfg=tcfg, model_params=model_params)

    # Advance the model
    # Convergence to the target score of the sampling strategy
    traj.advance(t_end=10.0, termination_criteria=[TargetScoreConvergenceCriterion(cfg.get("ams", "target_score"))])

    x_range = np.linspace(-1.6, 1.6, 101)
    y_range = np.linspace(-1.6, 2.6, 132)
    potential = np.zeros((101, 132))
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            potential[i, j] = fmodel.potential(np.array([x_range[i], y_range[j]]))

    state_x = np.array([s[1][0] for s in traj.get_state_list()])
    state_y = np.array([s[1][1] for s in traj.get_state_list()])

    X, Y = np.meshgrid(x_range, y_range)

    with contextlib.suppress(Exception):
        plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    plt.figure(figsize=(6, 4))
    ctr = plt.contourf(X, Y, np.transpose(potential), levels=25, cmap="viridis_r")
    plt.plot(state_x, state_y, color="k", alpha=0.8)
    plt.xticks(fontsize="x-large")
    plt.yticks(fontsize="x-large")
    plt.xlabel("x", fontsize="x-large")
    plt.ylabel("y", fontsize="x-large")
    plt.colorbar(ctr, label="V(x,y)")
    plt.grid(linestyle=":", color="silver")
    plt.tight_layout()
    plt.show()
