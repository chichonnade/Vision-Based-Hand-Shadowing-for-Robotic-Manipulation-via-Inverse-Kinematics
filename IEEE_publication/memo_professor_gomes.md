# Memo: Professor Gomes's Feedback — Resolution Summary

**Paper**: Vision-Based Hand Shadowing for Robotic Manipulation via Inverse Kinematics  
**Manuscript #**: Access-2026-11582  

---

## Abstract
| Request | Status |
|---------|--------|
| Define "IK" on first use | Done — "inverse-kinematics (IK)" defined in line 3 of abstract |
| Use "pipeline" consistently (not "system") | Done — "IK retargeting pipeline" used throughout |
| Rephrase 9.3% sentence ("and find that success is reduced to 9.3% due to...") | Done |

## Related Work
| Request | Status |
|---------|--------|
| Remove code-level class/function names (Transformation[InT, OutT]) | Done — all code references removed |
| Merge Table 1 and Figure 1 | Done — replaced with single TikZ architecture diagram (Fig. 1) |

## Pipeline Overview (Section III)
| Request | Status |
|---------|--------|
| Create hardware diagram with angles (θ, t_z, t_y) | Done — TikZ schematic in Fig. 2 (right panel) with θ=50°, t_z=0.48m, t_y=0.049m |
| Remove robot.jpg, replace with diagram | Done — Fig. 2 now uses `hardware_description.png` + TikZ schematic |

## Methods (Section IV)
| Request | Status |
|---------|--------|
| Remove "pyrealsense2 SDK" | Done — replaced with "RealSense SDK" |
| Define P_cam = (X, Y, Z) | Done — Eq. 1 defines P_cam explicitly |
| Eq. 2: Define P_t^raw, distinguish (u_i^raw, v_i^raw) vs (u_i, v_i) | Done |
| Depth: Use D[u_i, v_i] consistently | Done |
| Depth in metres (not scaled) | Done — "D (in metres)" |
| Use P_cam instead of [X,Y,Z] | Done |
| Remove [d_min, d_max] | Done |
| Replace THUMB_MCP/INDEX_FINGER_MCP with P_1, P_2 | Done |
| Use R, t instead of R_final, t_final | Done |
| Define θ in hardware diagram | Done — Fig. 2 shows θ arc |
| Define (p_target, q_target) | Done — Section IV.E |
| R = [e1 e2 e3] → q_target | Done — "quaternion formed by [e1 e2 e3]" |
| Remove scipy.spatial.transform reference | Done |
| Explain why d̂ uses unit vectors | Done — "normalising before averaging ensures both fingers contribute equally regardless of measured 3D lengths" |
| Add gripper vectors diagram | Done — Fig. 3 (TikZ) |
| Use "regularized least-squares problem" phrasing | Done — "by solving the following damped-least-squares (DLS) problem" |
| Replace "position p_target" with "pose (p_target, q_target)" | Done |
| Remove "PyBullet's calculateInverseKinematics" | Done — "PyBullet IK engine" |
| Mention solver iterations/threshold | Done — "100 iterations with residual threshold 10^-4" |
| Simplify gripper fallback explanation | Done — plain-language description with "depth fallback mechanism" |
| Remove g = 9.81 m/s² | Done |
| Explain PID tuning rationale | Done — "empirically tuned to match the motion speed and response characteristics of the physical robot arm" |

## Figures
| Request | Status |
|---------|--------|
| Figure 5 → tile matrix with numbered positions | Done — TikZ grid with tiles #1–#9, box, and robot (Fig. 6 left) |
| Add robotmountedcamera_pov.jpg | Done — Fig. 6 right panel |
| Make tile #6–#9 exclusion understandable | Done — matrix layout makes spatial reasoning clear |

## Experimental Setup (Section V)
| Request | Status |
|---------|--------|
| Remove software version list | Done |

## Author List
| Request | Status |
|---------|--------|
| Remove Yu-Chun Tuan biography | Done |
| Add Professor Gabriel Gomes as coauthor (top + bio) | Done — listed as author with affiliation and biography |

---

## Items for Your Review

1. **Hardware diagram (Fig. 2)**: The right panel is a TikZ schematic — please verify the geometry matches the physical setup.
2. **Gripper vectors diagram (Fig. 3)**: New TikZ figure showing e1, d, P1, P2, p_target — please verify correctness.
3. **Tile grid (Fig. 6)**: Verify numbering matches the physical layout from the robot's perspective.
4. **Joint variable renamed**: Changed from θ to **q** for joint angles to avoid ambiguity with camera tilt angle θ.
