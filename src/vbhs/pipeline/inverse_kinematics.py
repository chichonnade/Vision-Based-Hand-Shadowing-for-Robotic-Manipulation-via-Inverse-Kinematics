"""Inverse kinematics solver using PyBullet."""

# pylint: disable=c-extension-no-member

import logging
from typing import List, Optional

import pybullet as p

_logger = logging.getLogger(__name__)


class IKSolver:
    """Solver for inverse kinematics using PyBullet."""

    def __init__(self,
                 robot_id: int,
                 end_effector_idx: int,
                 joint_indices: List[int]):
        """ Initialize the IK solver.

        Args:
            robot_id: PyBullet robot body ID.
            end_effector_idx: Link index of the end effector.
            joint_indices: List of joint indices for IK.
        """
        self._robot_id = robot_id
        self._end_effector_idx = end_effector_idx
        self._joint_indices = joint_indices

        fixed_joint_indices = [
            i for i in self._joint_indices
            if p.getJointInfo(self._robot_id, i)[2] == p.JOINT_FIXED]
        if fixed_joint_indices:
            raise ValueError(
                (f'Joints {fixed_joint_indices} are fixed. '
                 'Only movable joints can be controlled.'))

        num_joints = p.getNumJoints(self._robot_id)
        all_joints = [i for i in range(num_joints)]
        all_movable_joints = [
            i for i in all_joints
            if p.getJointInfo(self._robot_id, i)[2] != p.JOINT_FIXED]

        # Index of controlled joints in the IK result.
        self._result_joint_indices = [all_movable_joints.index(i) for i in self._joint_indices]
        self._joint_damping = [
            float(p.getJointInfo(self._robot_id, joint_idx)[6])
            for joint_idx in all_joints]
        # Replace zero damping with small positive value to avoid IK solver crashes.
        self._joint_damping = [max(d, 0.001) for d in self._joint_damping]

    def solve(self,
              target_pos: tuple[float, float, float],
              target_orientation_quat: Optional[tuple[float, float, float, float]]
              ) -> Optional[list[float]]:
        """ Calculate joint angles for the target pose.

        Args:
            target_pos: Target 3D position [x, y, z].
            target_orientation_quat: Target orientation as quaternion [x, y, z, w].

        Returns:
            List of joint angles or None if IK fails.
        """

        joint_angles_tuple = self._calculate_ik(target_pos, target_orientation_quat)
        if joint_angles_tuple is None:
            _logger.warning("Inverse kinematics failed to find a solution")
            return None
        result_angles = [joint_angles_tuple[i] for i in self._result_joint_indices]
        assert len(result_angles) == len(self._joint_indices), (
            f'Expected {len(self._joint_indices)} angles, got {len(result_angles)}')
        return result_angles

    def _calculate_ik(
            self,
            target_pos: tuple[float, float, float],
            target_orientation_quat: Optional[tuple[float, float, float, float]]
            ) -> Optional[list[float]]:
        """Internal IK calculation using PyBullet."""
        num_joints = p.getNumJoints(self._robot_id)

        # Get the current joint states to use as the base for rest poses.
        joint_states = p.getJointStates(self._robot_id, range(num_joints))
        rest_poses = [state[0] for state in joint_states]

        target_pos_list = list[float](target_pos)
        target_orientation_quat_list = list[float](
            target_orientation_quat) if target_orientation_quat is not None else None

        joint_angles_tuple = p.calculateInverseKinematics(
            bodyUniqueId=self._robot_id,
            endEffectorLinkIndex=self._end_effector_idx,
            targetPosition=target_pos_list,
            targetOrientation=target_orientation_quat_list,
            restPoses=rest_poses,
            jointDamping=self._joint_damping,
            maxNumIterations=100,
            residualThreshold=1e-4)

        return list[float](joint_angles_tuple) or None

    def reset(self):
        """Reset the solver state."""
        # No state to reset anymore
