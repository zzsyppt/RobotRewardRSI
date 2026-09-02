from true_rl_agent.candidates.components import RewardInputs, RewardTerm, TrainingComponents

def audit_reward(inp: RewardInputs):
    return inp.base_height.square()

def build_training_components() -> TrainingComponents:
    return TrainingComponents(
        name='deterministic_audit_candidate',
        reward_terms=[
            RewardTerm(name='audit_reward', weight=1.0, fn=audit_reward),
        ],
        rationale='Deterministic dry-run audit mechanism.',
    )
