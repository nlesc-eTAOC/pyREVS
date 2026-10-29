import typing
from typing import Any
import numpy as np
from pyrevs.core import ForwardModelBaseClass


class OrnsteinUhlenbeck(ForwardModelBaseClass):
    """A one dimensional Ornstein-Uhlenbeck process.

    The classic Ornstein-Uhlenbeck process SDE:
        dX_t = -theta X_t dt + sqrt(2*epsilon) dW_t

    The stationary distribution of X_t is a Normal distribution
    with sigma = sqrt(epsilon/theta)

    This is the simple SDE used in Lestang et al. to
    develop and test TAMS.
    """

    def _init_model(self, m_id: int, params: dict[typing.Any, typing.Any]) -> None:
        """Concrete class specific initialization.

        Args:
            m_id: the model instance unique identifier
            params: an optional dict containing parameters
        """
        # Theta: the inverse time scale
        self._theta = params.get("theta", 1.0)

        # The noise parameter
        self._epsilon = params.get("epsilon", 0.5)

        # The resulting standard deviation of the process
        # distribution
        self._sigma = np.sqrt(self._epsilon / self._theta)

        # Store the time scale
        self._tau = 1.0 / self._theta

        # The model state is a float
        # drawn from the stationary distribution
        self._state = np.random.default_rng().normal(scale=self._sigma)

        # Initialize the RNG
        if self._deterministic:
            self._rng = np.random.default_rng(m_id)
        else:
            self._rng = np.random.default_rng()

    def _advance(self, _step: int, _time: float, dt: float, noise: Any, _need_end_state: bool) -> float:
        """Concrete class advance function.

        This is the model-specific advance function.

        Args:
            step: the current step counter
            time: the starting time of the advance call
            dt: the time step size over which to advance
            noise: the noise to be used in the model step
            need_end_state: whether the step end state is needed
        Return:
            Some model will not do exactly dt (e.g. sub-stepping) return the actual dt
        """
        # Advance using Euler-Maruyama explicit scheme
        noise_scaling = float(np.sqrt(2.0 * self._epsilon * dt))
        self._state = (1.0 - self._theta * dt) * self._state + noise_scaling * noise
        return dt

    def get_current_state(self) -> float:
        """Return the current state of the model.

        Note that the return type is left to the concrete model definition.
        """
        return self._state

    def set_current_state(self, state: float) -> None:
        """Set the current state of the model.

        Args:
            state: the externally provide state
        """
        self._state = state

    def score(self) -> float:
        """Return the model's current state score.

        The score is a real.

        Returns:
            the score associated with the current model state
        """
        return self._state

    def make_noise(self) -> Any:
        """Return the model's latest noise increment.

        Note that the noise type is left to the concrete model definition.

        Returns:
            The model next noise increment
        """
        # Return a builtin float: numpy scalars are slow to pickle
        return float(self._rng.standard_normal())

    @classmethod
    def name(cls) -> str:
        """Return a the model name."""
        return "OrnsteinUhlenbeck"
