"""Task-neutral robot execution axis."""

from true_rl_agent.robots.base import (
    BASE_ROLE,
    FORBIDDEN_CONTACT_ROLE,
    JOINT_GROUP_ROLE_PREFIX,
    SUPPORT_EFFECTOR_ROLE,
    RobotProfile,
    RobotRequirements,
    RobotRuntimeAdapter,
    SemanticRole,
    SemanticRoleRequirement,
    semantic_roles_are_mutually_exclusive,
)
from true_rl_agent.robots.registry import RobotRegistry

__all__ = [
    "BASE_ROLE",
    "FORBIDDEN_CONTACT_ROLE",
    "JOINT_GROUP_ROLE_PREFIX",
    "SUPPORT_EFFECTOR_ROLE",
    "RobotProfile",
    "RobotRegistry",
    "RobotRequirements",
    "RobotRuntimeAdapter",
    "SemanticRole",
    "SemanticRoleRequirement",
    "semantic_roles_are_mutually_exclusive",
]
