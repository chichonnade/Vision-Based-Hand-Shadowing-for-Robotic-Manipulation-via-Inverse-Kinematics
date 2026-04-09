# IEEE Paper Revision Plan

Comprehensive revision of `IEEE_publication/paper.tex` to address all IEEE reviewer comments (4 reviewers, ~30 unique concerns) and UC Berkeley professor feedback (~25 specific edits), incorporating existing experiment datasets and new metrics.

---

## Revision Tasks (17 items)

| # | Task | Status | Depends on Data? |
|---|------|--------|-------------------|
| 1 | Revise abstract | Pending | No |
| 2 | Strengthen introduction | Pending | No |
| 3 | Revise related work | Pending | No |
| 4 | Create TikZ hardware diagram | Pending | No |
| 5 | Fix notation across Methods | Pending | No |
| 6 | Create gripper vectors diagram | Pending | No |
| 7 | Revise IK/gripper/simulation text | Pending | No |
| 8 | Add hand detection reliability section | Pending | No (data exists) |
| 9 | Add end-effector accuracy section | Pending | **Yes (D1)** |
| 10 | Add sim-to-real transfer section | Pending | **Yes (D5)** |
| 11 | Add EMA ablation section | Pending | **Yes (D3)** |
| 12 | Add action distribution figure | Pending | No (data exists) |
| 13 | Revise results & discussion | Pending | **Yes (D4)** |
| 14 | Create TikZ tile grid diagram | Pending | No |
| 15 | Expand conclusion & future work | Pending | No |
| 16 | Update author list | Pending | No |
| 17 | Draft Response to Reviewers | Pending | Yes (all) |

---

## I. Abstract (lines 65-86)

- **Define IK on first use**: "...inverse kinematics (IK) pipeline..." (Prof)
- **Consistent terminology**: Use "pipeline" throughout, not interchangeably with "system" (Prof). Audit full paper.
- **Rephrase "complexity of mapping human hand articulations"** for clarity (R4.12)
- **Change 9.3% wording**: Replace "where hand occlusion by surrounding objects reduces success to 9.3% (N=75)" with "and find that success is reduced to 9.3% due to hand occlusion by surrounding objects" (Prof)

## II. Introduction (lines 98-133)

- **Stronger novelty positioning** (R1.1, R2.1, R3.1, R4.4): Add 1-2 paragraphs explicitly differentiating from existing teleop frameworks (exoskeleton-based, VR-based) and imitation learning. Key differentiator: zero-training, single RGB-D camera, no GPU, no specialized hardware, analytical rather than learned.
- **Cite recent teleop literature** (R4.13): Add references to AnyTeleop, H2O, Dex-Cap, and other 2024-2025 marker-free teleoperation works.
- **Highlight research gaps** (R4.4): Explicitly state that no prior work provides a complete RGB-D-to-IK pipeline for low-cost arms with zero training data.
- **Frame offline processing as a design choice** (R1.2, R2.2, R3.2): Emphasize this is an *offline retargeting and data-collection* tool, not a real-time controller.

## III. Related Work (lines 136-178)

- **Remove code references**: Delete the `Transformation[InT, OutT]` / `_transform` sentence from Section III (Prof). This appears at line 189-191.
- **Add a "Marker-Free Teleoperation" subsection** (R2.4, R3.5, R4.5): Compare with AnyTeleop, H2O, Dex-Cap, BundleSDF-based approaches. Clearly distinguish our pipeline's unique aspects.
- **Merge Table 1 and Figure 1** (Prof): Combine the TikZ architecture diagram with the pipeline stage table into a single figure, using the professor's sketched layout (labels A-H on the flow diagram with intermediate representations annotated directly). Remove the separate Table 1.

## IV. System Overview -- Hardware (lines 180-280)

- **Replace Figure 2 (robot.JPG)** with a TikZ schematic diagram (Prof): Create a hardware geometry diagram showing:
  - Camera mounting angle theta = 50 degrees
  - Camera height relative to robot base (translation offsets)
  - Gripper angles
  - Robot coordinate frame axes
- **Also include `hardware_description.png`** as a subfigure (Prof): Shows the annotated photo (RealSense D435i, angled mount, camera stand, 6-DOF arm, 12V battery).

## V. Methods -- Notation & Equation Fixes (lines 283-455)

### V.A. RGB-D Capture (lines 286-300)
- **Remove "pyrealsense2 SDK"** reference (Prof): Replace with "the RealSense SDK" or just "the camera driver."
- **Define (X, Y, Z)**: Introduce P_cam = (X, Y, Z)^T as the 3D point in camera coordinates (Prof).

### V.B. Hand Pose Estimation (lines 302-321)
- **Fix EMA equation (Eq. 2)** (Prof):
  - Define p_t^raw = (u_i^raw, v_i^raw) as the raw MediaPipe output
  - Define P_t^raw as the array of all raw landmarks at time t
  - Rewrite as: P_{t+1} = alpha * P_t^raw + (1-alpha) * P_t
  - Then "unpack" into (u_i, v_i)

### V.C. Depth-Based 3D Reconstruction (lines 323-346)
- **Use D[u_i, v_i]** (not D[v_i, u_i]) for consistent ordering (Prof)
- **Use P_cam** instead of [X, Y, Z] (Prof)
- **Remove [d_min, d_max] notation** since not used elsewhere; just state "valid range 0.1-5.0 m" (Prof)
- **Replace THUMB_MCP / INDEX_FINGER_MCP** with defined symbols P_1, P_2 (Prof). Define these landmark symbols once (with a reference to the MediaPipe hand topology) and use consistently throughout.

### V.D. Camera-to-Robot Transform (lines 348-386)
- **Use R and t** instead of R_final and t_final (Prof): Simplify notation throughout.
- **Define angle theta** in the hardware diagram from Section IV (Prof): Cross-reference the diagram.

### V.E. Target Pose Computation (lines 388-424)
- **Define (p_target, q_target)** explicitly (Prof)
- **R = [e1 e2 e3] should yield q_target**: State that the rotation matrix is converted to quaternion q_target (Prof). Remove `scipy.spatial.transform` reference.
- **Add gripper vectors diagram** (Prof): Include a figure based on `drafted_gripper_vectors.png` showing e1, e2, e3, d-hat, u_thumb, u_index vectors on the hand geometry.
- **Explain d-hat estimation** (Prof): Why d-hat = (u_thumb + u_index) / ||u_thumb + u_index|| instead of just d? Because in the general 3D case the average of the unit finger vectors gives a more robust pointing direction estimate than the raw difference, which can be noisy when landmarks are at different depths.
- **Replace THUMB_MCP etc.** with P_1, P_2 symbols (consistent with V.C)

### V.F. Inverse Kinematics (lines 426-455)
- **"by solving the following regularized least-squares problem"** before Eq. 8 (Prof)
- **Replace "position p_target"** with "pose (p_target, q_target)" (Prof)
- **Remove "PyBullet's calculateInverseKinematics"** function name (Prof). Instead write "The solver uses PyBullet's IK engine and runs up to 100 iterations..."
- **Add theoretical justification for damped-least-squares** (R4.6): Briefly cite Wampler (1986) or Nakamura & Hanafusa for DLS-IK.

### V.G. Gripper Control (lines 457-482)
- **Rename gripper angle** to "target gripper angle" or phi_target (Prof)
- **Simplify fallback explanation** (Prof): Rewrite as: "A depth fallback mechanism handles the critical case where exactly one of the two landmarks locating thumb and index finger is invalid. When the primary landmarks (fingertips) are unavailable, knuckle landmarks are used; when those also fail, the last valid angle is held; as a final default, the gripper is set to mid-open."
- **Add justification for fallback hierarchy** (R1.7, R3.9): Frame as a graceful degradation strategy common in sensor fusion systems; cite related robustness patterns.

### V.H. Simulation Preview (lines 484-517)
- **Remove "with gravity g = 9.81 m/s^2"** (Prof)
- **Replace PID description**: Instead of listing force/position/velocity gains, explain that "PID parameters were empirically tuned to match the motion speed and behavior of the physical robot arm" (Prof). Cross-reference the new sim-to-real tracking section.

## VI. New Experimental Results Sections

These are entirely new additions to address the largest reviewer gaps.

### VI.A. New Section: "Hand Detection Reliability" (R1.7, R1.8, R3.4)
**Data source**: `new experiment data (hand detection statistics)/`
- **Table from hand detection statistics**: Raw vs. corrected detections (503 frames)
  - Right-only: 77.3% -> 80.1%
  - Erroneous left swaps: 5.6% -> 2.8%
  - No hands: 17.1%
- Discuss: The hand-swap correction halves erroneous classifications. The 17.1% no-detection rate correlates with the failure modes in structured/unstructured environments.

### VI.B. New Section: "End-Effector Accuracy" (R1.8, R3.4)
**Data source**: Awaiting user data (D1)
- Cartesian distance between FK(actual_joints) and p_target per frame
- Angular error between achieved and target orientation
- Report: mean, std, max over representative trajectories
- This is the single most impactful addition for reviewers.

### VI.C. New Section: "Trajectory Smoothness" (R3.4)
**Data source**: Awaiting user data (D2), using .npy files
- Jerk metric (d^3 theta / dt^3) from .npy action files
- Report per-joint smoothness metrics
- Compare raw IK output vs. EMA-smoothed output

### VI.D. New Section: "Sim-to-Real Transfer Analysis" (R3.8)
**Data source**: `new experiment data (PID tuning sim vs real)/` + awaiting D5
- Include run 4 tracking error plot (best-tuned PID)
- Report per-joint RMSE between commanded and achieved positions
- Discuss: error oscillates around 0 (unbiased), wrist_flex and gripper are most challenging joints

### VI.E. New Section: "EMA Smoothing Ablation" (R3.7)
**Data source**: Awaiting user data (D3)
- Test with alpha in {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}
- Report success rate or tracking quality for each
- Show that alpha = 0.8 (landmarks) and alpha = 0.5 (IK) are near-optimal

### VI.F. New Figure: "Action Distribution Analysis" (R3.10)
**Data source**: `new expirement data (actions histograms)/`
- Include the motor-space action histogram
- Discuss joint limit saturation, especially shoulder_lift clipping
- Quantify percentage of frames where each joint hits its limits

### VI.G. Updated Success Rate Table with Confidence Intervals (R1.6)
**Data source**: Awaiting user data (D4)
- User will re-run experiment 2-3 times
- Report mean +/- std across runs

## VII. Results & Discussion Revisions (lines 680-910)

### VII.A. Latency Discussion (R1.2, R2.2, R3.2)
- Expand the latency section to discuss:
  - Optimization strategies: batch processing, GPU-accelerated MediaPipe, replacing overlay visualization (110ms) which is purely diagnostic
  - The 110ms overlay is a debugging visualization, not required for deployment -- removing it would bring latency to ~103ms (~10 FPS)
  - Frame why offline is acceptable: this is a data collection and trajectory generation tool, not a real-time controller

### VII.B. In-the-Wild Discussion (R1.3, R2.3, R3.3)
- Strengthen the 90% -> 9.3% drop discussion:
  - Use hand detection statistics to quantify detection failure rate
  - Discuss concrete mitigation strategies: temporal Kalman filtering, multi-view fusion, learned depth completion
  - Cite specific works on occlusion-robust hand tracking

### VII.C. VLA Comparison Fairness (R1.6, R2.4)
- Add a paragraph acknowledging the differences in training configurations
- Add statistical CIs to Table IV (from re-run data)
- Frame IK as a zero-shot baseline rather than a competing method

### VII.D. Expanded Failure Modes
- Integrate hand detection statistics into the failure mode discussion
- Add quantitative error breakdown (from new Cartesian error data)

## VIII. Conclusion & Future Work (lines 912-949)

- **Add future research direction**: Real-time optimization (removing overlay, GPU MediaPipe, CUDA IK) (R2.2, R4.11)
- **Add**: Multi-robot / higher-DOF applicability discussion (R3.11)
- **Add**: Advanced grasp estimation (6-DoF grasp pose) (R2.5)
- **Add**: Dynamic hand tracking with temporal models (R4.11)
- **Expand occlusion section**: More concrete -- name specific approaches (temporal transformers, multi-view stereo)

## IX. Figure Replacements

| Figure | Current | Replacement |
|--------|---------|-------------|
| Fig 1 + Table 1 | Separate TikZ diagram + pipeline table | Merged single figure with annotated flow (Prof sketch) |
| Fig 2 | `robot.JPG` | TikZ hardware geometry diagram + `hardware_description.png` |
| Fig 5 (tile grid) | `chart_pertile.png` | TikZ tile matrix + `robotmountedcamera_pov.jpg` |
| New | -- | Gripper vectors diagram (e1, e2, e3, d-hat) |
| New | -- | Sim-to-real tracking error plot (run 4) |
| New | -- | Action histograms in motor space |

## X. Author List Changes (Prof)

- **Remove**: Yu-Chun Tuan (Prof's bio text references this name, but the current paper has "Trevor Rigoberto Martinez" -- need to clarify if Trevor = Yu-Chun Tuan or if Trevor stays)
- **Add**: Professor Gabriel Gomes as co-author (Department of Mechanical Engineering, UC Berkeley). Add biography at end.
- Also requires the IEEE "Request for Byline Change" form from the `IEEE request/` folder.
- Update `\author{}` block and `\address[]` lines accordingly.

## XI. Response to Reviewers Document

After all paper changes are complete, draft the IEEE Access "Response to Reviewers" document mapping each reviewer comment to: (a) the concern, (b) our response, (c) the specific action taken in the manuscript. Use the template from `IEEE request/_ IEEE-Access-Response-to-Reviewers-template-6.26.23.docx`.

---

## Data Deliverables TODO (for the user)

These 5 items can be produced in parallel and are ordered by impact. Text-only edits (Sections I-V, VII-X) can proceed immediately without this data. New experimental sections (VI) will be drafted with placeholders and filled in once data arrives.

### D1. Cartesian End-Effector Error (HIGHEST PRIORITY)

**Addresses**: R1.8, R2.4, R3.4 (positional error, orientation error)

**What to compute**: Using the .npy action files + .bag files:
- For each non-NaN frame in block 1 (right arm), run FK(theta) using PyBullet to get the achieved end-effector position and orientation
- Compare against the stored p_target for that frame (from the pipeline's intermediate output)

**Deliverables** (6 numbers per trajectory, or aggregated across all files):
- Position error: mean (mm), std (mm), max (mm) of ||FK(theta) - p_target||
- Orientation error: mean (deg), std (deg), max (deg) of angular difference between FK quaternion and q_target

**Format**: A small table or CSV:
```
pos_mean_mm, pos_std_mm, pos_max_mm, ang_mean_deg, ang_std_deg, ang_max_deg
```

**Note**: If p_target is not stored in the .npy files, re-run the pipeline on the corresponding .bag files and log p_target alongside the IK output.

### D2. Trajectory Smoothness

**Addresses**: R3.4 (trajectory smoothness metric)

**What to compute**: From the .npy files (block 1, right arm):
- Filter out NaN frames (use only contiguous non-NaN segments)
- dt = 1/5 (since pipeline runs at ~5 FPS)
- Compute jerk for each joint: `jerk = np.diff(joint_angles, n=3, axis=0) / dt**3`
- RMS jerk per joint = `sqrt(mean(jerk**2))`
- **Do this twice**: (a) on the raw IK output, (b) on the EMA-smoothed output (alpha=0.5)

**Deliverables**: A table with 6 rows (one per joint) and 2 columns (raw RMS jerk, smoothed RMS jerk)

**Format**: Example row:
```
shoulder_pan, 45.2, 18.7  (units: rad/s^3)
```

### D3. EMA Smoothing Ablation

**Addresses**: R3.7 (parameter sensitivity / ablation study)

**What to compute**: Re-run the pipeline on a subset of the benchmark (e.g., 10 episodes: tiles #1-#5, 2 grasps each) with the following landmark EMA alpha values: {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}
- Keep IK EMA alpha fixed at 0.5

**Deliverables**: A table with 6 rows, columns:
```
alpha, success_count_out_of_10, notes
```

Also useful: mean end-effector jitter (std of frame-to-frame position change) for each alpha.

### D4. Re-Run Success Rates

**Addresses**: R1.6 (statistical significance and variance)

**What to compute**: Run the full 50-episode benchmark (tiles #1-#5, 10 grasps per tile) 2 more times (3 runs total including the original)

**Deliverables per run**: Per-tile success counts:
```
Run 1 (original): [10, 10, 9, 9, 7] = 45/50
Run 2: [?, ?, ?, ?, ?] = ?/50
Run 3: [?, ?, ?, ?, ?] = ?/50
```

From these I'll compute: mean success rate, std, and 95% CI.

### D5. Sim-to-Real RMSE

**Addresses**: R3.8 (sim-to-real gap quantification)

**What to compute**: From the PID tuning run 4 data:
- If you have the raw tracking error data (not just the plot), compute per-joint RMSE: `sqrt(mean(error**2))` for all 6 joints

**Deliverables**: 6 numbers in motor-space units [-100, 100]:
```
shoulder_pan_rmse, shoulder_lift_rmse, elbow_flex_rmse, wrist_flex_rmse, wrist_roll_rmse, gripper_rmse
```

If you only have the plot and not raw data, I can reference the plot qualitatively.

---

## Available Existing Data Summary

### Hand Detection Statistics (`new experiment data (hand detection statistics)/`)
- 503-frame detection table: raw vs. corrected classifications
- Ready to use as-is for new Table in paper

### PID Tuning / Sim vs Real (`new experiment data (PID tuning sim vs real)/`)
- 4 runs with progressive PID tuning
- Run 4 = best-tuned (highest P/D, removed accel/speed limits)
- Per-joint tracking error plots available
- Ready to use as figure; RMSE numbers needed (D5)

### Action Histograms (`new expirement data (actions histograms)/`)
- Raw action distributions in radians
- Motor-space action distributions
- Shows joint limit saturation (shoulder_lift clipping)
- Ready to use as figure

### Pick-and-Place Action Files (`new experiment data (pick and place action files)/`)
- 4 .npy files: (T, 2, 6) float32 joint angles in radians
- human_demo_actions_1: 608 frames (366 valid right-arm)
- human_demo_actions_2: 397 frames (205 valid right-arm)
- human_demo_actions_3: 413 frames (112 valid right-arm)
- human_demo_actions_4: 453 frames (172 valid right-arm)
- Useful for D2 (trajectory smoothness)

### Paired .bag + Actions (`new experiment data (realsense bag file...)`)
- human_demo.bag (raw RGB-D recording)
- human_demo_actions.npy: 274 frames (166 valid right-arm)
- Useful for D1 (can re-run pipeline to extract p_target)

---

## Reviewer Comment Cross-Reference

Below maps each reviewer comment to where it is addressed in the plan.

### Reviewer 1
| # | Comment Summary | Plan Section |
|---|----------------|--------------|
| 1 | Novelty not distinguished | II, III |
| 2 | 5 FPS too slow | II, VII.A |
| 3 | 90% to 9.3% drop | VII.B |
| 4 | No occlusion handling | VII.B, VIII |
| 5 | Narrow eval scope | VII.C, VIII |
| 6 | VLA comparison unfair | VII.C, VI.G |
| 7 | MediaPipe + heuristic dependence | VI.A, V.G |
| 8 | No error propagation / metrics | VI.B, VI.C |

### Reviewer 2
| # | Comment Summary | Plan Section |
|---|----------------|--------------|
| 1 | Relies on existing tools | II, III |
| 2 | 213ms latency, need 30 FPS | VII.A, VIII |
| 3 | 9.3% in-the-wild, no occlusion fix | VII.B, VIII |
| 4 | Only pick-and-place, no diverse benchmarks | VII.C, VIII |
| 5 | Gripper control too simple | V.G, VIII |

### Reviewer 3
| # | Comment Summary | Plan Section |
|---|----------------|--------------|
| 1 | Not distinguished from existing frameworks | II, III |
| 2 | ~5 FPS contradicts teleop | VII.A |
| 3 | 90% to 9.3% drop, no occlusion solution | VII.B |
| 4 | No metrics beyond success rate | VI.B, VI.C, VI.D |
| 5 | Incomplete SOTA comparison | III, VII.C |
| 6 | Only simple pick-and-place | VII.C, VIII |
| 7 | No ablation study | VI.E |
| 8 | Sim-to-real gap not quantified | VI.D |
| 9 | Fallback lacks theoretical justification | V.G |
| 10 | SO-ARM101 limitations not analyzed | VI.F, VIII |
| 11 | Multi-arm / higher-DOF not explored | VIII |

### Reviewer 4
| # | Comment Summary | Plan Section |
|---|----------------|--------------|
| 1-3 | Explain egocentric camera, 90%, VLA comparison | II, VII.C |
| 4-5 | Research gaps, novelty of IK retargeting | II, III |
| 6 | DLS-IK solver details | V.F |
| 7 | Gripper fallback hierarchy | V.G |
| 8-9 | More unstructured tests, more baselines | VII.B, VII.C, VIII |
| 10 | Deployment challenges | VII.B, VIII |
| 11 | Future research directions | VIII |
| 12 | Rephrase abstract | I |
| 13 | Cite recent teleop literature | II |
| 14 | Methodological clarity | V (all) |

### Professor
| Topic | Plan Section |
|-------|--------------|
| Define IK, consistent terminology | I |
| 9.3% wording | I |
| Remove code references | III |
| Merge Table 1 + Figure 1 | III |
| Hardware diagram with angles | IV |
| Replace robot.JPG | IV, IX |
| Notation fixes (Pcam, EMA, D[], P1/P2, R/t) | V.A-V.E |
| Gripper vectors diagram | V.E, IX |
| IK wording, remove PyBullet function name | V.F |
| Gripper fallback simplification | V.G |
| Simulation: remove gravity, explain PID | V.H |
| Tile grid figure | IX, XIV |
| Remove software version list | VII (Experimental Setup) |
| Author changes (add Gomes, remove Tuan) | X |
