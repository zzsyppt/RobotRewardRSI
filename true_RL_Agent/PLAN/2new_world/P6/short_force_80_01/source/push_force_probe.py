"""Bounded existing-checkpoint external-force diagnostic; no training or qualification."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from push_force_scoring import force_outcome, force_sequence_outcome
from push_sequence_scoring import raw_conditions
from true_rl_agent.composition import TaskCompositionContext
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider
from true_rl_agent.tasks.push_recovery.seed_policy import derive_runner_seed

CHECKPOINTS = {
    'go2': 'sha256:1495390b192b913432b8472cb150e2db07fdc2eb811eeabe4ab87c577b24e0ff',
    'g1': 'sha256:91c00d4e61492dcfb6511aa0ae9ef85121c5bb08dafde7eabf29c10281ad1749',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def prepare(args):
    provider = PushRecoveryCompositionProvider()
    values = {**provider.cli_defaults(), 'execution_device': args.device,
              'isaac_python': sys.executable, 'reference_program_id': 'push_standing_recovery_stationary_reference_v3'}
    if args.robot == 'go2':
        source = ROOT / 'outputs/p6_go2_matched_dev_01'
        record = read(source / 'result.json')
        values.update(**record['budget'], cell_variant_id=record['cell'],
                      push_protocol_path=str(Path(__file__).parent / 'calibration_protocols_01/scale_2p0.json'))
        formal = source / 'composition'
        session = 'p6-go2-matched2-dev-20260912'
        base_seed = 42
        historical = read(source / 'public_dev_result.json')['task_evidence']['seed_summaries']
        old_rows = [s['per_episode'] for s in historical]
        old_file = source / 'public_dev_result.json'
    else:
        source = ROOT / 'outputs/p6_g1_stationary_v3_qualification_20260912'
        formal = source
        contract = read(source / 'reference_qualification_execution.json')
        session = contract['session_id']
        base_seed = 1000000000
        values.update(cell_variant_id='push.g1_37dof.recovery_v2', eval_base_seed=base_seed,
                      runner_timeout_s=contract['runner_timeout_s_per_job'],
                      push_protocol_path=str(ROOT / 'configs/tasks/push_g1_37dof_recovery_v3.json'))
        artifact = read(source / 'reference_eval_artifact.json')
        ref = next(r for r in artifact['evaluation_assembly']['phase_receipts'] if r['phase'] == 'public_dev')
        old_file = Path(ref['path'])
        assert file_sha256(old_file) == ref['checksum']
        receipt = read(old_file)
        # Receipt layout is inspected in CPU preflight before any physical evaluation.
        old_rows = [s['per_episode'] for s in receipt['phase_result']['task_evidence']['seed_summaries']]
    train_path = formal / 'runner' / session / 'train/summary.json'
    train = read(train_path)
    checkpoint = Path(train['checkpoint']['metadata']['local_path'])
    assert train['checkpoint']['checksum'] == file_sha256(checkpoint) == CHECKPOINTS[args.robot]
    composition = provider.resolve(values=values, context=TaskCompositionContext(
        project_root=ROOT, output_dir=formal, read_only=True))
    manifest = composition.resolved_cell_manifest
    assert manifest.manifest_hash == train['resolved_cell_manifest_hash']
    old_env = read(source / 'backend_execution_environment.json')
    composition.backend_adapter.validate_session_execution_environment(execution_environment=old_env)
    current = composition.backend_adapter.resolve_session_execution_environment(
        session_id='p6-force-' + args.robot, session_dir=args.output, allow_create=True)
    assert current['manifest_hash'] == old_env['manifest_hash']
    write(args.output / 'backend_execution_environment.json', current)
    projection = manifest.robot_task_binding['task_projection']
    p = deepcopy(projection['push_resolved_protocol'])
    dt = projection['push_abi_facts']['control']['control_dt_s']
    pulse_steps = math.ceil(p['time_scale_s'] / dt)
    starts = [math.ceil(t / dt) for t in p['push_times_s']]
    window = math.ceil(p['recovery_window_s'] / dt)
    assert starts[0] + pulse_steps + window < starts[1]
    if args.short_repeat_sequence:
        starts[1] = starts[0] + 2 * pulse_steps
    assert starts[1] + pulse_steps + window < math.ceil(p['episode_length_s'] / dt)
    body_name = manifest.resolved_roles['base'][0]
    definition = {'schema_version': 'true_rl_agent.p6.external_force_diagnostic.v2',
        'robot': args.robot, 'mechanism': 'finite_world_horizontal_force_at_bound_base_link_com',
        'source_resolved_protocol_hash': p['resolved_protocol_hash'], 'body_name': body_name,
        'mass_rule': 'actual_all_robot_link_masses_sum_after_reset_per_environment',
        'force_rule': 'force_scale * total_mass_kg * source_delta_v_world_mps / actual_pulse_duration_s',
        'force_scale': args.force_scale,
        'positions': None, 'direct_torque_nm': [0, 0, 0], 'is_global': True,
        'control_dt_s': dt, 'physics_dt_s': projection['push_abi_facts']['control']['simulation_dt_s'],
        'pulse_control_steps': pulse_steps, 'pulse_duration_s': pulse_steps * dt,
        'start_control_steps': starts, 'window_steps': window,
        'dwell_steps': math.ceil(p['stable_dwell_s'] / dt),
        'first_recovery_sample': 'pulse_end_control_step_plus_one',
        'recovery_scope': 'every_completed_pulse_independent_window_then_final_stable',
        'base_seed': base_seed, 'derived_seeds': [derive_runner_seed(base_seed=base_seed, split_name='public_dev', run_index=i) for i in range(1, 4)],
        'envs_per_seed': 64, 'seed_count': 3, 'train_jobs': 0, 'holdout_jobs': 0, 'provider_calls': 0,
        'wall_limit_s': 1800, 'comparison_scope': 'source_velocity_result_is_historical_different_mechanism'}
    if args.short_repeat_sequence:
        definition.update(schema_version='true_rl_agent.p6.external_force_diagnostic.v3',
            recovery_scope='all_planned_pulses_then_one_fresh_window_and_final_stable',
            repeat_gap_rule='ceil(time_scale_s / control_dt_s) * control_dt_s',
            repeat_gap_control_steps=pulse_steps,
            repeat_gap_s=pulse_steps * dt,
            comparison_scope='G_2x_long_gap_raw_rescored_with_same_sequence_rule')
        p['push_times_s'] = [s * dt for s in starts]
    if args.repeated_high_fraction is not None:
        definition.update(schema_version='true_rl_agent.p6.external_force_diagnostic.v4',
            repeated_high_fraction=args.repeated_high_fraction,
            repeated_source_delta_v_rule='source_high_delta_v_mps * repeated_high_fraction',
            comparison_scope='H_2x_short_gap_60pct_same_sequence_rule')
    definition['definition_hash'] = canonical_json_hash(definition)
    p.update(schema_version='true_rl_agent.p6.force_parameters.v1', mechanism=definition['mechanism'],
             source_protocol_hash=p.pop('protocol_hash'), protocol_hash=definition['definition_hash'])
    p.pop('resolved_protocol_hash')
    p['resolved_protocol_hash'] = canonical_json_hash(p)
    write(args.output / 'parameters.json', p)
    write(args.output / 'definition.json', definition)
    write(args.output / 'historical_dev_episodes.json', old_rows)
    identity = {'checkpoint_sha256': CHECKPOINTS[args.robot], 'source_manifest_hash': manifest.manifest_hash,
        'source_training_binding_hash': train['training_binding_hash'], 'source_train_summary_sha256': file_sha256(train_path),
        'source_dev_file_sha256': file_sha256(old_file), 'source_environment_hash': current['manifest_hash'],
        'definition_hash': definition['definition_hash'], 'script_sha256': file_sha256(__file__),
        'scoring_sha256': file_sha256(Path(__file__).with_name('push_force_scoring.py'))}
    write(args.output / 'source_identity.json', identity)
    return composition, checkpoint, p, definition, identity


def run(args, composition, checkpoint, parameters, definition):
    import numpy as np
    import torch
    import warp as wp
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from true_rl_agent.backends.isaac_lab_arrays import as_torch
    from true_rl_agent.backends.rsl_rl_checkpoint import load_inference_checkpoint
    from true_rl_agent.robots.rsl_rl_runtime import rsl_rl_agent_configuration
    from true_rl_agent.tasks.push_recovery.runtime import (physical_runtime_config_from_manifest,
        instantiate_physical_push_configs, make_physical_push_environment, push_policy_io_evidence, push_runtime_state)
    from true_rl_agent.tasks.push_recovery.rollout import _finite

    manifest = composition.resolved_cell_manifest
    config = physical_runtime_config_from_manifest(manifest, 'strong_push')
    launch = SimpleNamespace(seed=definition['base_seed'], num_envs=64, device=args.device)
    torch.manual_seed(launch.seed)
    env_cfg, agent_cfg = instantiate_physical_push_configs(config, launch, evaluation=True)
    env_cfg.events.push_velocity_increment = None
    env_cfg.push_protocol_parameters = deepcopy(parameters)
    video_options = None
    if args.video_env_index is not None:
        from true_rl_agent.runner.native_video import NativeVideoOptions
        video_options = NativeVideoOptions(args.output / 'video', env_index=args.video_env_index,
            resolution=(1280, 720), eye=(2.3, -2.3, 1.5), lookat=(0.0, 0.0, 0.0))
        video_options.configure_viewer(env_cfg)
    base = make_physical_push_environment(env_cfg, render_mode='rgb_array' if video_options else None)
    wrapped = None
    try:
        assert not base._physics_handles_decimation
        abi = push_policy_io_evidence(base, manifest)
        agent_cfg.device = args.device
        wrapped = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
        learner = rsl_rl_agent_configuration(config['runtime_adapter_id']).verify_runner(runner)
        loaded = load_inference_checkpoint(runner, str(checkpoint))
        write(args.output / 'runtime_admission.json', {'policy_io': abi, 'learner': learner, 'checkpoint_loading': loaded})
        policy = runner.get_inference_policy(device=base.device)
        if video_options:
            from true_rl_agent.runner.native_video import NativeVideoRecorder, _encode_frames
            # Reuse camera/clone visibility setup, but sample before reset below.
            recorder = NativeVideoRecorder(base, video_options)
        state, robot = push_runtime_state(base), base.scene['robot']
        body_ids, body_names = robot.find_bodies(definition['body_name'], preserve_order=True)
        assert body_names == [definition['body_name']]
        body_index = body_ids[0]
        ids_tensor = torch.tensor(body_ids, device=base.device, dtype=torch.int32)
        cpu = lambda v: as_torch(v).detach().cpu().numpy().copy()
        all_rows, seed_summaries = [], []
        for seed_index, seed in enumerate(definition['derived_seeds'], 1):
            if args.video_seed_index is not None and seed_index > args.video_seed_index:
                break
            # Reproduce the original reset/rollout history before the selected seed.
            capture_video = video_options is not None and seed_index == args.video_seed_index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            base.seed(seed)
            state.begin_evaluation('public_dev', seed)
            reset = wrapped.reset()
            obs = reset[0] if isinstance(reset, tuple) else reset
            video_frames = []
            if capture_video:
                recorder.start(f'seed_{seed_index:02d}_env_{args.video_env_index:02d}', seed)
            masses = as_torch(robot.data.body_mass).clone()
            total_mass = masses.sum(dim=1)
            assert bool(torch.isfinite(masses).all()) and bool((total_mass > 0).all())
            initial_xy, origins = cpu(state.initial_xy), cpu(base.scene.env_origins)
            strata, planned = cpu(state.stratum), cpu(state.planned)
            magnitude = torch.full((64,), parameters['velocity_magnitudes_mps']['medium'], device=base.device)
            magnitude[state.stratum == 1] = parameters['velocity_magnitudes_mps']['low']
            magnitude[state.stratum == 3] = parameters['velocity_magnitudes_mps']['high']
            if 'repeated_high_fraction' in definition:
                magnitude[state.stratum == 4] = (parameters['velocity_magnitudes_mps']['high']
                                                * definition['repeated_high_fraction'])
            direction = torch.stack((torch.cos(state.angle), torch.sin(state.angle), torch.zeros_like(state.angle)), dim=-1)
            expected_force = direction * (definition['force_scale'] * total_mass * magnitude / definition['pulse_duration_s'])[:, None, None]
            frames, physics = [], []
            pending = {}
            original_write, original_apply = robot.write_data_to_sim, robot.root_view.apply_forces_and_torques_at_position
            original_compute = base.termination_manager.compute

            def apply_wrench(*a, **kw):
                force_b = wp.to_torch(kw['force_data']).reshape(64, robot.num_bodies, 3)
                torque_b = wp.to_torch(kw['torque_data']).reshape(64, robot.num_bodies, 3)
                assert kw['position_data'] is None and kw['is_global'] is False
                physics.append({**pending, 'delivered_force_b': cpu(force_b[:, body_index]),
                    'delivered_torque_b': cpu(torque_b[:, body_index]),
                    'body_quaternion_xyzw': cpu(robot.data.body_link_quat_w)[:, body_index],
                    'body_com_position_w': cpu(robot.data.body_com_pos_w)[:, body_index],
                    'other_body_force_max': np.array(float(force_b[:, [i for i in range(robot.num_bodies) if i != body_index]].abs().max()))})
                return original_apply(*a, **kw)

            def inject_write():
                step = as_torch(base.episode_length_buf)
                event = torch.full((64,), -1, device=base.device, dtype=torch.long)
                force = torch.zeros((64, 3), device=base.device)
                for j, start in enumerate(definition['start_control_steps']):
                    active = (~state.closed) & (state.planned > j) & (step >= start) & (step < start + definition['pulse_control_steps'])
                    force[active] = expected_force[active, j]
                    event[active] = j
                pending.clear()
                pending.update(control_step_before=cpu(step), valid=cpu(~state.closed), event_index=cpu(event), requested_force_w=cpu(force))
                robot.instantaneous_wrench_composer.set_forces_and_torques_index(
                    forces=force[:, None, :], torques=torch.zeros_like(force[:, None, :]),
                    positions=None, body_ids=ids_tensor, is_global=True)
                return original_write()

            def capture_compute():
                frames.append({'valid': cpu(~state.closed), 'steps': cpu(base.episode_length_buf),
                    'position': cpu(robot.data.root_pos_w), 'quaternion_xyzw': cpu(robot.data.root_quat_w),
                    'linear': cpu(robot.data.root_lin_vel_w), 'angular': cpu(robot.data.root_ang_vel_w)})
                if capture_video and bool(frames[-1]['valid'][args.video_env_index]):
                    rgb = base.render().copy()
                    assert rgb.shape == (video_options.resolution[1], video_options.resolution[0], 3), rgb.shape
                    video_frames.append(rgb)
                return original_compute()

            robot.write_data_to_sim = inject_write
            robot.root_view.apply_forces_and_torques_at_position = apply_wrench
            base.termination_manager.compute = capture_compute
            try:
                for _ in range(int(base.max_episode_length)):
                    with torch.inference_mode():
                        actions = policy(obs)
                    assert _finite(obs) and _finite(actions)
                    result = wrapped.step(actions)
                    obs = result[0]
                    assert _finite(obs) and _finite(result[1])
                    if bool(state.closed.all()):
                        break
            finally:
                robot.write_data_to_sim = original_write
                robot.root_view.apply_forces_and_torques_at_position = original_apply
                base.termination_manager.compute = original_compute
                robot.instantaneous_wrench_composer.reset()
            assert bool(state.closed.all())
            assert len(physics) == len(frames) * int(base.cfg.decimation)
            arrays = {key: np.stack([r[key] for r in frames]) for key in frames[0]}
            physical = {key: np.stack([r[key] for r in physics]) for key in physics[0]}
            assert all(np.isfinite(v).all() for v in arrays.values())
            assert all(np.isfinite(v).all() for v in physical.values())
            # Lab3 body quaternion is xyzw: rotate the actually delivered local force to world.
            q = physical['body_quaternion_xyzw']
            v = physical['delivered_force_b']
            cross = 2 * np.cross(q[..., :3], v)
            delivered_w = v + q[..., 3:4] * cross + np.cross(q[..., :3], cross)
            errors = np.linalg.norm(delivered_w - physical['requested_force_w'], axis=-1)
            assert float(errors.max()) < 1e-3
            assert float(np.abs(physical['delivered_torque_b']).max()) < 1e-5
            assert float(physical['other_body_force_max'].max()) < 1e-5
            raw_path = args.output / f'seed_{seed_index:02d}.npz'
            np.savez_compressed(raw_path, **arrays, **{'physics_' + k: v for k, v in physical.items()},
                                body_masses_kg=cpu(masses), expected_force_w=cpu(expected_force))
            rows = []
            for env_id in range(64):
                valid = arrays['valid'][:, env_id]
                steps = arrays['steps'][valid, env_id].tolist()
                stable, failure = raw_conditions(**{k: arrays[k][valid, env_id] for k in ('position', 'quaternion_xyzw', 'linear', 'angular')},
                    initial_xy=initial_xy[env_id], origin=origins[env_id], parameters=parameters)
                events = []
                for j, start in enumerate(definition['start_control_steps']):
                    mask = physical['event_index'][:, env_id] == j
                    count = int(mask.sum())
                    if not count:
                        continue
                    expected_substeps = definition['pulse_control_steps'] * int(base.cfg.decimation)
                    event_steps = physical['control_step_before'][mask, env_id]
                    events.append({'event_index': j, 'start_control_step': int(event_steps[0]),
                        'end_control_step': int(event_steps[-1]) + 1, 'actual_substeps': count,
                        'expected_substeps': expected_substeps, 'completed': count == expected_substeps,
                        'injection_verified': bool(np.all(errors[mask, env_id] < 1e-3)),
                        'force_world_n': cpu(expected_force)[env_id, j].tolist(),
                        'delivered_impulse_world_ns': (delivered_w[mask, env_id].sum(axis=0) * base.physics_dt).tolist()})
                scorer = force_sequence_outcome if args.short_repeat_sequence else force_outcome
                outcome = scorer(steps, stable, events, int(planned[env_id]), failure_label=failure,
                    final_stable=stable[-1], window_steps=definition['window_steps'], dwell_steps=definition['dwell_steps'])
                rows.append({'env_id': env_id, 'stratum': ('control', 'low', 'medium', 'high', 'repeated')[int(strata[env_id])],
                    'episode_steps': steps[-1], 'planned_push_count': int(planned[env_id]), 'force_events': events, **outcome})
            assert not sum(r['invalid_injection_count'] for r in rows)
            record = {'seed': seed, 'definition_hash': definition['definition_hash'], 'raw_sha256': file_sha256(raw_path),
                'initial_xy': initial_xy.tolist(), 'origins': origins.tolist(), 'body_names': robot.body_names,
                'force_body_index': body_index, 'mass_range_kg': [float(total_mass.min()), float(total_mass.max())],
                'force_transform_max_error_n': float(errors.max()), 'physics_substeps': len(physics),
                'capture_point': 'control_state_after_physics_before_termination_and_reset', 'episodes': rows}
            write(args.output / f'seed_{seed_index:02d}.json', record)
            if capture_video:
                selected = rows[args.video_env_index]
                assert len(video_frames) == selected['episode_steps']
                video_path = video_options.output_dir / f'seed_{seed_index:02d}_env_{args.video_env_index:02d}.mp4'
                _encode_frames(video_path, video_frames, resolution=video_options.resolution, fps=1 / base.step_dt)
                import imageio_ffmpeg
                decoder = imageio_ffmpeg.read_frames(str(video_path), pix_fmt='rgb24')
                video_header = next(decoder)
                decoded_count = sum(1 for _ in decoder)
                assert decoded_count == len(video_frames)
                assert tuple(video_header['size']) == video_options.resolution
                write(args.output / 'video.json', {'file': str(video_path), 'sha256': file_sha256(video_path),
                    'seed_index': seed_index, 'seed': seed, 'env_index': args.video_env_index,
                    'frame_count': len(video_frames), 'fps': 1 / base.step_dt,
                    'resolution': list(video_options.resolution),
                    'decoded_frame_count': decoded_count,
                    'preceding_seed_rollouts_replayed': seed_index - 1,
                    'capture_point': 'after_physics_before_termination_and_reset',
                    'first_control_step': 1, 'last_control_step': selected['episode_steps'],
                    'selected_episode': selected, 'diagnostic_replay_only': True})
                video_frames.clear()
            all_rows.extend(rows)
            seed_summaries.append({'seed': seed, 'successes': sum(r['success'] for r in rows), 'episodes': 64})
            print(json.dumps({'robot': args.robot, **seed_summaries[-1]}), flush=True)
        summary = {'episodes': len(all_rows), 'successes': sum(r['success'] for r in all_rows), 'seeds': seed_summaries,
            'failure_counts': dict(Counter(r['failure_label'] for r in all_rows)),
            'strata': {s: {'episodes': sum(r['stratum'] == s for r in all_rows),
                'successes': sum(r['success'] for r in all_rows if r['stratum'] == s)} for s in ('low', 'medium', 'high', 'repeated')},
            'invalid_injections': sum(r['invalid_injection_count'] for r in all_rows)}
        write(args.output / 'summary.json', summary)
        return summary
    finally:
        (wrapped or base).close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', choices=('go2', 'g1'), required=True)
    parser.add_argument('--mode', choices=('preflight', 'run'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--expected-definition-hash')
    parser.add_argument('--force-scale', type=float, choices=(1.0, 1.5, 2.0), default=1.0)
    parser.add_argument('--short-repeat-sequence', action='store_true')
    parser.add_argument('--repeated-high-fraction', type=float, choices=(0.8,))
    parser.add_argument('--video-env-index', type=int, choices=range(64))
    parser.add_argument('--video-seed-index', type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    if args.repeated_high_fraction is not None and not args.short_repeat_sequence:
        parser.error('repeated high fraction belongs to the short sequence diagnostic')
    if (args.video_env_index is None) != (args.video_seed_index is None):
        parser.error('video requires both env and seed index')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'diagnostic_only': True, 'robot': args.robot, 'mode': args.mode,
        'started_at_unix': time.time(), 'train_jobs': 0, 'holdout_jobs': 0, 'provider_calls': 0}
    app, code = None, 1
    try:
        composition, checkpoint, p, definition, identity = prepare(args)
        report['source_identity'] = identity
        if args.expected_definition_hash:
            assert args.expected_definition_hash == definition['definition_hash']
        if args.mode == 'run':
            assert args.expected_definition_hash
            from isaaclab.app import AppLauncher
            from true_rl_agent.runner.isaac_app_lifecycle import bind_isaac_log_path
            launch = argparse.Namespace(headless=True, device=args.device, enable_cameras=args.video_env_index is not None)
            bind_isaac_log_path(launch, args.output / 'result.json')
            app = AppLauncher(launch, multi_gpu=False).app
            report['summary'] = run(args, composition, checkpoint, p, definition)
        report['status'] = 'PASS' if args.mode == 'run' else 'PREPARED'
        code = 0
    except Exception:
        report.update(status='FAIL', error=traceback.format_exc())
        traceback.print_exc()
    report['ended_at_unix'] = time.time()
    write(args.output / 'result.json', report)
    if app is not None:
        from true_rl_agent.runner.isaac_app_lifecycle import close_isaac_app_on_success
        code = close_isaac_app_on_success(app, exit_code=code)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
