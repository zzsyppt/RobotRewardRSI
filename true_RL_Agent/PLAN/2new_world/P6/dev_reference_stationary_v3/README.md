This is an independent P6 development hypothesis, not a qualified reference.

Reference ID: `push_standing_recovery_stationary_reference_v3`.
Complete program: `training_code/components.py`.

Only the old stationary term is replaced, with term ID `stationary_v3` and
weight 1.0. For v = norm(base_lin_vel_scaled[:, :2]) / 0.08 and
w = norm(base_ang_vel_scaled) / 0.5, the rate is
`1 - 0.25 * (H(v) + H(w))`, where `H(x) = 0.5*x*x` for x <= 1 and
`H(x) = x - 0.5` otherwise. The implementation uses q = min(x, 1),
`H(x) = 0.5*q*q + x - q`, which is equivalent for nonnegative norms and
works with the existing task smoke tensors and CPU Torch tensors.

The five-term container uses a list, as required by the existing static verifier;
the original reference used a tuple. Term order is unchanged.
The other four functions, term weights, inputs, and integration convention
are unchanged from standing_recovery_v2. This directory changes no runtime
defaults, protocol, termination, qualification threshold, PPO setting, or budget.

Run `validate_cpu.py` with the project Python environment. It verifies the exact
program bytes using the existing static verifier and task-owned smoke adapter,
then checks Torch CPU behavior for both robot dimensions. Its temporary candidate
package is a verifier input only; it is not a trained candidate or a Genesis.
The generated reference manifest binds source identity, not empirical qualification.
The fall check compares instantaneous reward rates only. It does not show that
deliberate early termination cannot improve full-episode or discounted return.
