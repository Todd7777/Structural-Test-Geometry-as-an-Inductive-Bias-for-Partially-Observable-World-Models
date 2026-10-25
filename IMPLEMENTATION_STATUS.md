# Implementation status

This code was written for the camera-ready release on 25 October 2026, after the paper was accepted. It is a reference implementation of the method, not the archived code that generated the accepted experiments. The manuscript's empirical results remain unverified against raw experiment records in this repository. New outputs must be reported as new experiments.

## Correspondence to the paper

| Component | This release |
| --- | --- |
| Structural-test pseudometric | Weighted total variation for categorical profiles and normalized one-dimensional Wasserstein distance for empirical scalar profiles |
| Posterior marginalization | Exact finite-state posterior filtering and simulator forks in the reference GridWorld; no realized hidden goal is supplied to TR, MD, QP, RNC, or BD-S |
| QP | All unordered within-batch pairs, detached analytic scale, 0.99 EMA initialized at 1, clipping to [0.1, 10], denominator guard |
| MD | Symmetric `[zi + zj; zi * zj; abs(zi - zj)]`, 768→256→256→1 ReLU network at latent dimension 256; 262,913 parameters |
| Categorical TR | 256→128→output ReLU head, soft-target cross-entropy for each test |
| Continuous TR | Five-component Gaussian mixture, 1,000 soft Gumbel–Softmax samples at temperature 0.5, empirical W1; **hard sorting replaces the paper's strength-1 regularized SoftSort** |
| BD-S | Bivariate Gaussian coordinate NLL, categorical or factorized categorical cross-entropy, and five-component Gaussian-mixture NLL are available; the executable GridWorld pipeline uses the bivariate Gaussian |
| PH | Realized two-dimensional goal regression, used only when PH is selected |
| RNC | Multi-positive InfoNCE on raw latent states, distance threshold 0.2 and temperature 0.1; anchors without positives are ignored |
| Shared RSSM | Default 200-dimensional GRU and 56-dimensional Gaussian stochastic state; deployed latent concatenates the GRU state and posterior mean |
| CEM | Default horizon 10, population 200, 10 iterations, 20 elites; discounted predicted rewards plus discounted terminal value; no structural actions or latent-distance term in the planning score |
| IQD | Nonnegative analytic fitted scale on held-out pair distances; collapsed latent states have IQD 1 for nonzero target geometry |
| Statistics | Strict matched task/seed/episode IDs, shared seed resampling across fixed tasks, paired episode resampling within each task/seed |

## New reference choices

The paper does not specify enough detail to recover the original environment or every training convention from prose. The following choices are explicit here rather than inferred to be historical facts.

The reference GridWorld has a 7×7 wall border with a walkable 5×5 interior. The agent starts at (3, 3). Actions are no-op, x−, x+, y−, and y+; a move into a wall stays in place. The observation is `(x/6, y/6, current_reward)`. Goal coordinates are (1, 1), (1, 5), (5, 1), and (5, 5). Reaching the goal yields reward 1 and terminates; otherwise reward is 0. Episodes end after 50 actions. Success means reaching the goal within that horizon. This success rule belongs to the reference environment and does not establish the original six-task success rules.

Training draws goals from the first three corners and uses the corresponding three-corner inference prior. ID evaluation draws from those same three corners; shift evaluation draws the fourth corner. Both evaluation target filters use a uniform prior over all four corners, so the split does not reveal the realized goal to inference. The world model and planner receive only observations and ordinary actions, never the posterior, split-specific goal, or probes.

The 100-test library follows the paper's six-anchor/five-continuation plus 70-random-variant layout. Anchors are four corner teleports, a centre teleport, and no probe. Each anchor has three random and two coordinate-greedy continuations. The remaining tests use random continuations. Continuation lengths are sampled once from 3–7 actions with a fixed test seed. Greedy policies have fixed target corners; they cannot inspect the sampled hidden goal. Forks restart the continuation clock. A goal hit reveals its corner; otherwise the four-category terminal sensor emits a uniformly random corner label. This sensor is a new bounded response kernel, not a claim about the original simulator's sensor. Teleports are available only to structural target construction.

The default response estimator uses 100 posterior-sampled rollouts and a 0.5 pseudocount per category. `--exact` instead integrates the posterior and continuation randomness by finite-state dynamic programming, without smoothing. The exact mode changes target estimation and is identified in cache metadata.

The collector records one randomly selected history prefix from each independent random-policy episode and stores the full episode for base-model replay. Initial previous actions are zero vectors; subsequent actions are one-hot. The base loss is the sum of observation MSE, current-reward MSE, future-return MSE, and weighted posterior-to-prior Gaussian KL. Value targets are discounted Monte Carlo returns from the recorded random policy, excluding the reward already observed at the current step. These are reference value targets, not recovered details of the paper's value-learning procedure. Posterior samples feed recurrent transitions during base training; prediction heads and auxiliary losses use the posterior-mean latent. Training is offline on the collected cache. There is no online replay-refresh loop or million-transition rolling buffer.

The auxiliary sampler clusters flattened response vectors with seeded k-means and samples across occupied clusters with replacement. The separate base batch samples complete recorded episodes uniformly. Losses average over valid observations; padding does not change history encodings. Reference defaults use 128-unit encoder and prediction hidden layers, unit KL weight, gradient clipping at 100, and Adam with learning rate 3×10⁻⁴ and betas (0.9, 0.999). The small smoke configuration reduces model size, batches, and updates and is unsuitable for scientific comparisons.

The paper defines IQD only for nonzero target second moment. The library returns `(IQD=0, scale=0)` for all-zero targets as an explicit extension, and diagnostics also save the target second moment. A zero target geometry is uninformative for comparing representations.

## Not included

- Original experiment source, checkpoints, raw episode outcomes, or verified confidence intervals.
- The original six-task simulators and their exact success definitions.
- Key–Box, Hidden-Mass CartPole, Visual GridWorld, MiniWorld–Room, and MiniGrid–HiddenKey integrations.
- The visual VAE, 1,000-particle visual inference, calibration oracle, TD-MPC architecture check, and manuscript ablation suite.
- The paper's complete hyperparameter sweep, 10-seed terminal experiments, or compute measurements.
- A claim that hard-sorted continuous TR is equivalent to regularized SoftSort.

## Verification

The test suite checks analytic metric cases, pseudometric invariants, IQD invariance and collapse, strict bootstrap pairing, model gradients, exact MD size, Gaussian likelihoods, masking, CEM behavior, posterior filtering, response integration, no realized-state leakage into targets, and training/checkpoint loading for all seven implemented objectives. Deployment and held-out diagnostics are also tested with QP. Separate small command-line runs exercise training, ordinary-action planning, held-out IQD, and paired bootstrap output. These checks verify implementation behavior; they do not verify the accepted empirical results or establish full-training performance.

Tested on CPU with Python 3.12.14; exact dependency versions are in [requirements-tested.txt](requirements-tested.txt). Reproducibility across hardware and library versions is not guaranteed. The checkpoint and logs record the actual runtime, configuration, cache metadata, and dataset hash.
