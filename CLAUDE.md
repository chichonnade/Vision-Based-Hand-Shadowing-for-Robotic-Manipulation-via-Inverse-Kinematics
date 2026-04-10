# Claude Code Context

## Current Work: IEEE Paper Revision (paper-revision-occlusion-mitigation branch)

### What's Been Done
1. **paper.tex**: All professor requests addressed (27/27), ~75% of IEEE reviewer concerns addressed
2. **paper.tex**: "system" -> "pipeline" consistency fix (3 occurrences)  
3. **paper.tex**: Added paragraph reframing offline 5 FPS processing as a feature for generating imitation-learning action labels from pre-recorded RGB-D video
4. **scripts/compare_hand_detectors.py**: Comparison script run on `data/human_demo_short.bag` (347 frames). Results: MediaPipe 62.8% detection / 59.9% valid IK target; WiLoR 67.7% / 64.8% (+7.8% / +8.2%). WiLoR recovered 12 frames (3.5%) that MediaPipe missed.
5. **wilor_hand_detector.py**: Fixed handedness bug -- WiLoR's MANO `is_right` was in facing-perspective convention (same as MediaPipe's raw labels), so it now swaps for egocentric (FPV) view, matching MediaPipe's swap logic.
6. **paper.tex**: Added Section "Occlusion Mitigation via WiLoR" (after Joint Saturation Analysis) with Table comparing MediaPipe vs WiLoR on 347-frame recording
7. **paper.tex**: Updated Related Work (II.A) to cross-reference WiLoR occlusion evaluation
8. **paper.tex**: Updated Hand Detection Reliability section with WiLoR cross-reference
9. **paper.tex**: Updated In-the-Wild Evaluation to reference occlusion mitigation section
10. **paper.tex**: Updated Future Work occlusion bullet to note WiLoR is implemented/evaluated
11. **paper.tex**: Updated Abstract to mention WiLoR occlusion mitigation
12. **paper.tex**: Verified no remaining "system" references to the authors' pipeline

### What's Left
- **Remaining IEEE reviewer concerns** not yet addressed (see REVISION_PLAN.md tasks 8-15 for experimental additions that may need user-provided data: D1-D5)
- **Compile paper.tex** to verify no LaTeX errors (no LaTeX installed on this Mac)
- **Draft response-to-reviewers document** (REVISION_PLAN.md task 17)

### Key Files
- `IEEE_publication/paper.tex` -- the paper being revised
- `scripts/compare_hand_detectors.py` -- WiLoR vs MediaPipe comparison script
- `IEEE request/email.md` -- IEEE reviewer comments (4 reviewers)
- `Professor request/REQUEST.MD` -- professor feedback
- `REVISION_PLAN.md` -- original revision plan (predates this session)
- `src/vbhs/pipeline/hands/wilor_hand_detector.py` -- WiLoR detector (handedness fixed)
- `src/vbhs/pipeline/hands/mediapipe_hand_detector.py` -- MediaPipe detector

### Conda Environment
The conda env is called `env` (not `vbhs`): `conda activate env`

### Bag Files for Testing
- `data/human_demo_short.bag` -- short recording (347 frames, 1.7 GB) used for comparison
