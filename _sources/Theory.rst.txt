.. highlight:: rst

.. _sec:theory:

Theory
======

Stochastic dynamical systems
----------------------------

Stochastic dynamical systems describe the evolution of a random process :math:`(X_t)_{t \ge 0}`,
where the state :math:`X_t \in \mathbb{R}^{d}` evolves according to probabilistic rules
rather than deterministic dynamics only. If the system is Markovian, i.e.
:math:`X_{t+\Delta t}` depends only on the current state :math:`X_t`, its behavior can be
characterized by a stochastic differential equation (SDE) of the form:

.. math::

   dX_t = f(X_t)\,dt + g(X_t)\,dW_t.

where :math:`f : \mathbb{R}^{d} \to \mathbb{R}^{d}` denotes the *drift* term,
governing the deterministic component of the dynamics, while
:math:`g : \mathbb{R}^{d} \to \mathbb{R}^{d \times m}` represents the
*diffusion* matrix, which scales the stochastic forcing introduced by the
:math:`m`-dimensional Wiener process :math:`W_t \in \mathbb{R}^{m}`.

Solving analytically the above SDE to obtain the system probability distributions is
rarely feasible, especially in nonlinear or high-dimensional settings. Markov
chain Monte Carlo (MCMC) methods address this challenge by constructing a
discrete-time Markov chain whose stationary distribution approximates the
target distribution of the stochastic dynamical system. In this
context, MCMC produces an ensemble of sample trajectories
:math:`\{X_t^{(k)}\}_{k=1}^K`, where each :math:`X_t^{(k)}` represents the
:math:`k`-th simulated path of the underlying Markov process. These sampled
trajectories make it possible to approximate expectations :math:`\mathbb{E}[h(X_t)]`,
estimate uncertainty, and analyze long-term behavior even when closed-form expressions
for the dynamics are inaccessible.

Rare events
-----------

Rare events of the above stochastic dynamical system correspond to outcomes of the
process :math:`X_t` that occur with very low probability but often carry
significant impact, such as extreme fluctuations in financial markets, system
failures, climate extremes or transitions between metastable states in physical systems.
Let us denote such a rare event by

.. math::

   \mathcal{E} = \{ X_t \in \mathcal{C} \},

where :math:`\mathcal{C}` is a subset of the state space :math:`\mathbb{R}^{d}` with
small probability under the distribution of :math:`X_t`. Quantities of interest often
involve the expectation of an indicator function 

.. math::

   \mathbb{P}(\mathcal{E}) = \mathbb{E}[\mathbf{1}_{\mathcal{E}}(X_t)].

Estimating :math:`\mathbb{P}(\mathcal{E})` using plain Monte Carlo involves
drawing :math:`K` independent trajectories :math:`\{X_t^{(k)}\}_{k=1}^K` and
computing the empirical average, resulting in an estimator:

.. math::
  \hat{P}_K = \frac{1}{K} \sum_{k=1}^K \mathbf{1}_{\mathcal{E}}(X_t^{(k)}).

While unbiased, this estimator suffers from a severe relative error issue when the occurrence of
the event :math:`\mathcal{E}` is rare. Specifically, the relative error can be written as:
(`Rubino and Tuffin <https://onlinelibrary.wiley.com/doi/book/10.1002/9780470745403>`_):

.. math::

  \mathrm{RE}(\hat{P}_K)
    = \frac{\sqrt{\mathrm{Var}(\hat{P}_K)}}{\mathbb{P}(\mathcal{E})} ~
   \sim \mathcal{O}\!\left(
       \frac{1}{\sqrt{K\,\mathbb{P}(\mathcal{E})}}
     \right),

which implies that achieving a fixed relative accuracy requires
:math:`K = \mathcal{O}(1 / \mathbb{P}(\mathcal{E}))` samples. For rare events
where :math:`\mathbb{P}(\mathcal{E})` is extremely small, this makes plain
Monte Carlo computationally infeasible, motivating the use of variance
reduction and specialized rare-event sampling techniques such as importance
sampling, splitting, or rare-event-focused Markov chain Monte Carlo methods.

.. Note::
    In the following, we will focus on the rare event techniques
    originally implemented in `pyREVS`, other sampling methods
    might have been implemented afterward but it is up to developers to update the documentation.

Adaptive multilevel splitting
-----------------------------

Adaptive Multilevel Splitting (AMS), introduced by `Cerou and Guyader <https://www.tandfonline.com/doi/abs/10.1080/07362990601139628>`_,
is a modern rare-events sampling technique of the Importance Splitting (IS) family. The idea of splitting methods
is to divide the occurrence of a rare event into a chain of not-as-rare events, the probability of which is easier
to estimate. In contrast with earlier versions of the method, which required defining this sequence of not-as-rare
events a-priori, AMS allows to define it dynamically as the algorithm proceeds. For a historical perspective on AMS
see for instance `Cerou et al. <https://doi.org/10.1063/1.5082247>`_.

Using the previously introduced notations, let's define a state :math:`\mathcal{A}`,
a subset of :math:`\mathbb{R}^{d}`, and the associated return time to :math:`\mathcal{A}`:

.. math::

  \tau_{\mathcal{A}} = \inf \{t \in \mathbb{N} : X_t \in \mathcal{A}\}

for a Markov chain initiated at :math:`X_t(t=0) = X_0 = x_0 \notin \mathcal{A}`. Let's define similarly
:math:`\tau_{\mathcal{B}}` for a state :math:`\mathcal{B}`. When studying rare transitions, both :math:`\mathcal{A}` and :math:`\mathcal{B}`
are metastable regions of the system phase space if a Markov chain started in the vicinity of :math:`\mathcal{A}`
(resp. :math:`\mathcal{B}`) remains close to :math:`\mathcal{A}` (resp. :math:`\mathcal{B}`) for
a long time before exiting. In such cases, a rare transition from :math:`\mathcal{A}` to :math:`\mathcal{B}` can be defined as:


.. math::

   \mathbb{P}(E_{\mathcal{A}\mathcal{B}}) = \mathbb{P}(\tau_{\mathcal{B}} < \tau_{\mathcal{A}})
    
for initial conditions :math:`x_0` close to :math:`\mathcal{A}`. AMS was developed with rare transitions in mind,
but is not limited to transitions as rare occurrences of a model can also be sampled, i.e. :math:`\mathcal{B}` might
not be a metastable region.

The idea of multilevel splitting is to decompose this rare event into a series of `less rare` events by
defining successive regions :math:`\mathcal{C}_i`:

.. math::

    \mathcal{C}_I = \mathcal{B} \in \mathcal{C}_{I-1} \in ... \in \mathcal{C}_{1} \in \mathcal{C}_{0} = \mathcal{A}

and :math:`E_{\mathcal{A}\mathcal{C}_i}` the event :math:`\tau_{\mathcal{C}_i} < \tau_{\mathcal{A}}`.
Using the conditional probabilities :math:`p_i = \mathbb{P}(E_{\mathcal{A}\mathcal{C}_i} | E_{\mathcal{A}\mathcal{C}_{i-1}})`
for :math:`i > 1`, we have:

.. math::

    \mathbb{P}(E_{\mathcal{A}\mathcal{C}_I}) = \prod_{i=1}^I p_i

In practice, the definition of the regions in the state phase space :math:`\mathbb{R}^{d}` relies on
system observables, :math:`\mathcal{O}(X_t)`. These observables can be combined into a score function
:math:`\xi(X_t) : \mathbb{R}^{d} \to \mathbb{R}` mapping your high-dimensional state space to a more manageable
one dimensional space. The :math:`\mathcal{A}` and :math:`\mathcal{B}` states can then be defined using
:math:`\xi(X_t)`:

.. math::

    \mathcal{A} = \{X_t \in \mathbb{R}^{d} : \xi(X_t) < \xi_a \} \\
    \mathcal{B} = \{X_t \in \mathbb{R}^{d} : \xi(X_t) > \xi_b \}

The successive regions :math:`\mathcal{C}_i` can similarly be defined using levels of :math:`\xi` between
:math:`\xi_a` and :math:`\xi_b`. In AMS, these levels are automatically selected by the algorithm which alleviates
a strong convergence issue arising with older multilevel splitting methods which required selecting these levels
a-priori, using the practitioner intuition.

Trajectory-adaptive multilevel splitting
----------------------------------------

Trajectory-Adaptive Multilevel Splitting (TAMS) (`Lestang et al. <https://doi.org/10.1088/1742-5468/aab856>`_) is
a variant of AMS specifically designed to estimate rare events return time. In practice, it differs from AMS
in that the system is allowed to come back to its initial state :math:`\mathcal{A}`, but a finite time horizon 
for the rare event is set at :math:`T_a`. The sampled rare event is thus defined as:

.. math::

   \mathbb{P}(E_{\mathcal{B}}) = \mathbb{P}(\tau_{\mathcal{B}} < T_a)


TAMS algorithm
--------------

The idea behind AMS and TAMS is to iterate over a small ensemble of size :math:`N` of Markov chain trajectories (i.e. much
smaller than the number of trajectories needed for a reliable sampling of the rare transition event
with Monte Carlo), discarding trajectories that drift away from :math:`\mathcal{B}`, while
cloning/branching trajectories that are going towards :math:`\mathcal{B}`. This effectively biases the ensemble
toward the rare transition event.

The selection process uses the score function :math:`\xi(X_t,t)`. At each iteration :math:`j` of the algorithm,
the trajectories are ranked based on the maximum of :math:`\xi(X_t,t)`:

.. math::
   \mathcal{Q}_{tr} = \sup \; \; \xi(X_t,t)

for :math:`tr \in [1, N]` (for TAMS, :math:`t \in [0, T_a]`). At each iteration, the
:math:`l_j` trajectories with the smallest value of :math:`\mathcal{Q}`, :math:`min (\mathcal{Q}_{tr}) = \mathcal{Q}^*`,
are discarded and new trajectories are branched from the remaining trajectories and
advanced in time until they reach :math:`\mathcal{B}`, return to :math:`\mathcal{A}` (AMS)
or until the maximum time :math:`T_a` is reached (TAMS).
The process is illustrated on a small ensemble in the following figure:

.. figure:: images/TAMS_Illustration.png
   :name: TAMS branching
   :align: center
   :width: 90%

   Branching trajectory :math:`1` from :math:`3`, starting after :math:`\xi(X_3(t),t) > \mathcal{Q}^*`.

Note that at each iteration, selecting :math:`\mathcal{Q}^*` and discarding the :math:`l_j` lowest trajectories amounts
to defining a new :math:`\mathcal{C}_i` and defining :math:`p_i = 1 - l_j/N`.

For each cloning/branching event, a trajectory :math:`\mathcal{T}_{rep}` to branch from is selected
randomly (uniformly) in the :math:`N-l_j` remaining trajectories in the ensemble.
The branching time :math:`t_b` along :math:`\mathcal{T}_{rep}` is selected to ensure that the branched
trajectory has a score function strictly higher than the discarded one:

.. math::
   t_b = \inf \left\{t : \xi(X_{rep}(t), t) > \mathcal{Q}^* \right\}

This iterative process is repeated until all trajectories reached the measurable set :math:`\mathcal{B}` or
until a maximum number of iterations :math:`J` is reached. (T)AMS associates to the trajectories forming
the ensemble at step :math:`j` a weight :math:`w_j`:

.. math::
   w_j = \prod_{i=1}^{j} \left(1 - \frac{l_i}{N} \right) = \left(1 - \frac{l_j}{N} \right)w_{j-1}

Note that :math:`w_0 = 1`. The final estimate of :math:`p` is given by:

.. math::
  \hat{P} = \frac{N_{\in \mathcal{B}}^J}{N} \prod_{i=0}^J \left(1 - \frac{l_i}{N} \right)

where :math:`N_{\in \mathcal{B}}^J` is the number of trajectories that reached :math:`\mathcal{B}` at step :math:`J`.

(T)AMS only provides an estimate of :math:`p` and the algorithm is repeated several times in order to
get a more accurate estimate, as well as a confidence interval. The choice of :math:`\xi` is critical
for the performance of the algorithm as well as the quality of the estimator. Repeating the algorithm
:math:`K` times (i.e. performing :math:`K` TAMS runs) yields:

.. math::
   \overline{P}_K = \frac{1}{K}\sum_{k=1}^K \hat{P}_k

Theoretical analysis of the AMS method (which also extends to TAMS) have shown that the relative error
of :math:`\overline{P}_K` scales in the best case scenario:

.. math::
  \mathrm{RE}(\overline{P}_K)
   \sim \mathcal{O}\!\left(
       \sqrt{
           \frac{-\mathrm{log}(\mathbb{P}(E_{\mathcal{A}\mathcal{B}}))}
                {K}
            }
     \right),

while the worst case scenario is similar to plain Monte Carlo:

.. math::
  \mathrm{RE}(\overline{P}_K)
   \sim \mathcal{O}\!\left(
       \frac{1}{\sqrt{K\,\mathbb{P}(E_{\mathcal{A}\mathcal{B}})}}
     \right),

The best case scenario corresponds to the ideal case where the intermediate
conditional probabilities :math:`p_i` are perfectly computed, which corresponds to
using the optimal score function :math:`\overline{\xi}(y) = \mathbb{P}_y(\tau_{\mathcal{B}} < \tau_{\mathcal{A}})`,
also known as the `committor function`. One will note that the committor function is
precisely what the TAMS algorithm is after for :math:`y = X_0 = x_0`.

An overview of the algorithm is provided hereafter:

.. |nbsp| unicode:: 0xA0 0xA0 0xA0 0xA0 0xA0 0xA0
.. |nbsp2| unicode:: 0xA0 0xA0 0xA0 0xA0 0xA0 0xA0 0xA0 0xA0 0xA0 0xA0 0xA0 0xA0

.. raw:: html

   <blockquote>

1.   Simulate :math:`N` independent trajectories of the dynamical system (between [0, :math:`T_a`] for TAMS)
2.   Set :math:`j = 0` and :math:`w[0] = 1`
3.   while :math:`j < J`:
4.   |nbsp| compute :math:`\mathcal{Q}_{tr}` for all :math:`tr \in [1, N]` and sort
5.   |nbsp| select the :math:`l_j` smallest trajectories
6.   |nbsp| for :math:`i \ in [1, l_j]`:
7.   |nbsp2| select a trajectory :math:`\mathcal{T}_{rep}` at random in the :math:`N-l_j` remaining trajectories
8.   |nbsp2| branch from :math:`\mathcal{T}_{rep}` at time :math:`t_b` and advance :math:`\mathcal{T}_{i}` until it reaches :math:`\mathcal{B}`, :math:`\mathcal{A}` or :math:`T_a`
9.   |nbsp| set :math:`w[j] = (1 - l_j/N) \times w[j-1]`
10.  |nbsp| set :math:`j = j+1`
11.  |nbsp| if :math:`\mathcal{Q}_{tr} > \xi_{b}` for all :math:`tr \in [1, N]`:
12.  |nbsp2| break

.. raw:: html

   </blockquote>

Simple 1D double well
---------------------
.. _ssec:theory_1ddw:

To illustrate the above theory on a simple example, we now consider the concrete case
of a 1D double well stochastic dynamical system described by:

.. math::

   dX_t = f(X_t)\,dt + \sqrt{2\epsilon}\,dW_t.

where :math:`f(X_t) = - \nabla V(X_t)` is derived from a symmetric potential:

.. math::

   V(x) = \frac{1}{4}x^4 - \frac{1}{2}x^2,

:math:`\epsilon` is a noise scaling parameter, and :math:`W_t` a 1D Wiener process.

.. figure:: images/DoubleWell1D_intro.png
   :name: DoubleWell1D
   :align: center
   :width: 70%

   1D double well, showing the potential :math:`V` and distribution function of long Markov chains
   starting in each well, at two noise levels.

The figure above shows the two metastable states of the model: for long Markov chains
(:math:`T_f = N_t \delta_t` with :math:`N_t = 1e^6` and :math:`\delta_t = 0.01`) initiated
in either of the two wells (clearly marked by local minima of the potential :math:`V(x)`),
the model state distributions :math:`P(X_t)` remains in the well,
with the distribution widening as the noise level is increased.

Using TAMS, we want to evaluate the transition probability from :math:`\mathcal{A}`, the well
centered at :math:`x_{\mathcal{A}} = -1.0`, to :math:`\mathcal{B}` the well centered at
:math:`x_{\mathcal{B}} = 1.0`, with a time horizon :math:`T_a = 10.0` and setting :math:`\epsilon = 0.025`.

Even for a simple system like this one, there are multiple possible choices for the score function.
We will use a normalized distance to :math:`\mathcal{B}`:

.. math::

    \xi(X_t) = 1.0 - \frac{\Vert X_t - x_{\mathcal{B}}\Vert_2}{\Vert x_{\mathcal{A}} - x_{\mathcal{B}} \Vert_2}

and select :math:`\xi_b = 0.95`. Note that in this case, choosing a fixed value of :math:`\xi_a` is possible
but for simplicity, all the trajectories are initialized exactly at :math:`X_0 = x_{\mathcal{A}}`. Additionally,
no dependence on time is included in the score function (it is referred to as a `static` score function).
The animation below shows the evolution of the TAMS trajectories ensemble (each trajectory
:math:`\xi(X_t)` as function of time is plotted), during the course of the algorithm. As iterations progress,
the ensemble deviates from the system metastable state near :math:`\xi(X_t) = 0` towards higher values of :math:`\xi(X_t)`.

.. figure:: images/tams_doublewell1D.gif
   :name: TAMS_DoubleWell1D
   :align: center
   :width: 70%

   Evolution of the TAMS trajectories ensemble (:math:`N = 100`, :math:`l_j = 1`)
   over the course the algorithm iterations.

Eventually, all the trajectories transition to :math:`\mathcal{B}` and the algorithm stops.
The algorithm then provides :math:`\hat{P} = 4.2e^{-5}`. As mentioned above, the algorithm will need to be
performed multiple times in order to provide the expectation of the estimator :math:`\hat{P}`, and
comparing the relative error to the lower and upper bound mentioned above can be used to
evaluate the quality of the employed score function.
