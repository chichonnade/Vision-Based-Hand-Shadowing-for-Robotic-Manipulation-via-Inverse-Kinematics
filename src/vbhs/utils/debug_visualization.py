"""Debug visualization helpers.

This module defines the `DebugVisualizer` class which provides convenience
methods to render useful debugging aids in a PyBullet simulation:

* Target position + orientation triad with a small origin sphere
* Arm state – joint spheres, joint labels, and end-effector marker

The code was extracted from `robot_arm_estimation.py` so that visualization
logic is decoupled from the control/IK pipeline.
"""

from typing import Dict, List, Optional, cast
import enum

import pybullet as p


class DebugVisualizationMode(enum.Enum):
    """Mode for debug visualization."""
    OFF = 0
    TARGET_ONLY = 1
    BOTH = 2

    def next_mode(self) -> 'DebugVisualizationMode':
        """Get the next mode."""
        return DebugVisualizationMode((self.value + 1) % 3)


class DebugVisualizer:  # pylint: disable=too-many-instance-attributes
    """Light-weight wrapper around PyBullet debug rendering primitives."""

    def __init__(self, robot_id: Optional[int] = None, axis_len: float = 0.07):
        self.robot_id = robot_id
        self.axis_len = axis_len
        self._mode = DebugVisualizationMode.OFF

        # Containers keyed by arm ("left" / "right") holding debug item IDs
        self._debug_target_items: Dict[str, Dict[str, object]] = {}
        self._debug_arm_items: Dict[str, Dict[str, object]] = {}

    @property
    def mode(self) -> DebugVisualizationMode:
        """Get the current mode."""
        return self._mode

    def next_mode(self):
        """Get the next mode."""
        self._mode = self._mode.next_mode()
        if self._mode == DebugVisualizationMode.OFF:
            self.clear_all()

    def update_robot_id(self, robot_id: int):
        """Update internal robot reference (call if simulation reloads)."""
        self.robot_id = robot_id

    def visualize(self, arm: str, target_pos: Optional[tuple[float, float, float]],
                  orientation_quat: Optional[tuple[float, float, float, float]],
                  end_effector: int, arm_joints: list[int]):
        """Visualize the target and arm state."""
        match self._mode:
            case DebugVisualizationMode.OFF:
                return
            case DebugVisualizationMode.TARGET_ONLY:
                self._visualize_target(arm, target_pos, orientation_quat)
            case DebugVisualizationMode.BOTH:
                self._visualize_target(arm, target_pos, orientation_quat)
                self._visualize_arm_state(arm, end_effector, arm_joints)

    def _visualize_target(self, arm: str, target_pos: Optional[tuple[float, float, float]],
                          orientation_quat: Optional[tuple[float, float, float, float]]):
        """Render the target triad at *target_pos* with the supplied orientation."""
        if self.robot_id is None:
            return

        self._init_debug_target_visuals(arm)
        items = self._debug_target_items.get(arm)
        if items is None:
            return

        # Early return if target position is missing
        if target_pos is None:
            return

        # Move origin sphere & label (always show position if available)
        p.resetBasePositionAndOrientation(items["origin_body"], target_pos, [0, 0, 0, 1])  # type: ignore[index]
        p.addUserDebugText(f"target_{arm}", target_pos, textColorRGB=[1, 0, 0], textSize=1,
                        replaceItemUniqueId=items["target_text"])  # type: ignore[index]

        # Only draw orientation axes if orientation is available
        if orientation_quat is not None:
            # Rotate axes by quaternion
            rot_mat = p.getMatrixFromQuaternion(orientation_quat)
            x_axis = [rot_mat[0] * self.axis_len, rot_mat[1] * self.axis_len, rot_mat[2] * self.axis_len]
            y_axis = [rot_mat[3] * self.axis_len, rot_mat[4] * self.axis_len, rot_mat[5] * self.axis_len]
            z_axis = [rot_mat[6] * self.axis_len, rot_mat[7] * self.axis_len, rot_mat[8] * self.axis_len]

            # Update axes
            p.addUserDebugLine(target_pos,
                               [target_pos[0] + x_axis[0], target_pos[1] + x_axis[1], target_pos[2] + x_axis[2]],
                               [1, 0, 0], lineWidth=2, replaceItemUniqueId=items["x_line"])  # type: ignore[index]
            p.addUserDebugLine(target_pos,
                               [target_pos[0] + y_axis[0], target_pos[1] + y_axis[1], target_pos[2] + y_axis[2]],
                               [0, 1, 0], lineWidth=2, replaceItemUniqueId=items["y_line"])  # type: ignore[index]
            p.addUserDebugLine(target_pos,
                               [target_pos[0] + z_axis[0], target_pos[1] + z_axis[1], target_pos[2] + z_axis[2]],
                               [0, 0, 1], lineWidth=2, replaceItemUniqueId=items["z_line"])  # type: ignore[index]

    # --------------------------- Arm state -----------------------------
    def _visualize_arm_state(self, arm: str, end_effector: int, arm_joints: List[int]):
        """Render joint spheres and EE marker for *arm*."""
        if self.robot_id is None:
            return

        self._init_debug_arm_visuals(arm, arm_joints)
        items = cast(Dict[str, object], self._debug_arm_items.get(arm))
        if not items:
            return

        # End effector marker
        try:
            link_state = p.getLinkState(self.robot_id, end_effector)
            link_pos = link_state[4]
            p.resetBasePositionAndOrientation(items["ee_body"], link_pos, [0, 0, 0, 1])  # type: ignore[index]
            p.addUserDebugText(f"EE_{arm}", link_pos, textColorRGB=[0, 1, 0], textSize=1.2,
                               replaceItemUniqueId=items["ee_text"])  # type: ignore[index]
        except p.error:
            pass

        # Joint markers
        for i, joint_idx in enumerate(arm_joints):
            try:
                joint_info = p.getJointInfo(self.robot_id, joint_idx)
                link_state = p.getLinkState(self.robot_id, joint_idx)
                link_pos = link_state[4]
                p.resetBasePositionAndOrientation(items["joint_bodies"][i], link_pos, [0, 0, 0, 1])  # type: ignore[index]
                joint_name = joint_info[1].decode("utf-8")
                p.addUserDebugText(f"J{joint_idx}:{joint_name}", link_pos,
                                   textColorRGB=[0, 0, 1], textSize=0.8,
                                   replaceItemUniqueId=items["joint_texts"][i])  # type: ignore[index]
            except p.error:
                pass

    # --------------------------- Cleanup -------------------------------
    def clear_all(self):
        """Remove every debug item created by the visualizer."""
        for debug_dict in (self._debug_target_items, self._debug_arm_items):
            for arm_items in debug_dict.values():
                for key, item in arm_items.items():
                    if key.endswith("_body") or key.endswith("_bodies"):
                        bodies = item if isinstance(item, list) else [item]
                        for body_id in bodies:
                            try:
                                p.removeBody(body_id)
                            except p.error:
                                pass
                    else:
                        # All debug lines and texts fall here
                        debug_items = item if isinstance(item, list) else [item]
                        for dbg_id in debug_items:
                            try:
                                p.removeUserDebugItem(dbg_id)
                            except p.error:
                                pass
        self._debug_target_items.clear()
        self._debug_arm_items.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_debug_target_visuals(self, arm: str):
        if self.robot_id is None or arm in self._debug_target_items:
            return

        items: Dict[str, object] = {}

        # Origin sphere marking exact target position
        origin = [0, 0, -1]  # off-screen placeholder
        origin_sphere_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 0.8])
        items["origin_body"] = p.createMultiBody(baseMass=0, baseVisualShapeIndex=origin_sphere_shape,
                                                 basePosition=origin)

        # Axes debug lines
        axis_stub_len = 0.05
        items["x_line"] = p.addUserDebugLine(origin, [origin[0] + axis_stub_len, origin[1], origin[2]], [1, 0, 0], 2)
        items["y_line"] = p.addUserDebugLine(origin, [origin[0], origin[1] + axis_stub_len, origin[2]], [0, 1, 0], 2)
        items["z_line"] = p.addUserDebugLine(origin, [origin[0], origin[1], origin[2] + axis_stub_len], [0, 0, 1], 2)

        # Label
        items["target_text"] = p.addUserDebugText(f"target_{arm}", origin, textColorRGB=[1, 0, 0], textSize=1)

        self._debug_target_items[arm] = items

    def _init_debug_arm_visuals(self, arm: str, arm_joints: List[int]):
        if self.robot_id is None or arm in self._debug_arm_items:
            return

        items: Dict[str, object] = {}

        # End-effector sphere (green)
        ee_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 0.7])
        items["ee_body"] = p.createMultiBody(baseMass=0, baseVisualShapeIndex=ee_shape, basePosition=[0, 0, -1])
        items["ee_text"] = p.addUserDebugText(f"EE_{arm}", [0, 0, 0], textColorRGB=[0, 1, 0], textSize=1.2)

        # Joint spheres (blue) and labels
        joint_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.015, rgbaColor=[0, 0, 1, 0.7])
        items["joint_bodies"] = [
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=joint_shape, basePosition=[0, 0, -1])
            for _ in arm_joints
        ]
        items["joint_texts"] = [
            p.addUserDebugText(f"J{idx}", [0, 0, 0], textColorRGB=[0, 0, 1], textSize=0.8)
            for idx in arm_joints
        ]

        self._debug_arm_items[arm] = items
