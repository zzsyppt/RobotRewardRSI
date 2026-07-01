from humanoid_agent.code_sandbox.components import TrainingComponents


def build_training_components(task_spec, base_env_cfg, base_agent_cfg):
    _ = (task_spec, base_env_cfg, base_agent_cfg)
    return TrainingComponents(name="test_candidate")
