# Nazish Day-by-Day Task Tracker

## Purpose of this tracker
This tracker is for **Nazish only**. It turns the team plan into a strict execution plan so work does not drift.

Your lane is the **exploratory non-AMD lane**:
- first understand and run the baseline,
- then isolate one module,
- then attempt one backend path,
- then report feasibility and blockers clearly.

This tracker assumes the team baseline is the `torchbci` Kilosort-style pipeline on the **C46 dataset**, and that your side is the harder experimental track compared to the AMD track.

---

## Your role in one sentence
I am responsible for the **baseline understanding + one-module backend experiment + blocker analysis** for the non-AMD path.

---

## First decision you must settle before coding
Your team chat mixes **TensorRT** and **Tenstorrent**. These are not the same target.

Before deep implementation, send this message to the team:

> Please confirm my exact target for this week: NVIDIA TensorRT or actual Tenstorrent stack. I will proceed on one target only.

### If the answer is TensorRT
Your goal is:
- run baseline,
- isolate one module,
- attempt PyTorch -> ONNX -> TensorRT path,
- record unsupported ops / graph issues.

### If the answer is Tenstorrent
Your goal is:
- run baseline,
- isolate one module,
- attempt rewrite / compatibility study for TT-style execution,
- record required rewrites / unsupported ops / layout issues.

Do **not** try to support both in the same first pass.

---

# WEEK PLAN OVERVIEW

## End-of-week success criteria
By the end of this work block, you should have:
1. baseline run status,
2. pipeline map,
3. one standalone module test,
4. one backend attempt,
5. blocker list,
6. one concise team update.

## What does NOT count as your job this week
- full end-to-end porting,
- AMD implementation,
- adding a second algorithm,
- optimizing clustering,
- writing the whole team report.

---

# DAY 0 - ALIGNMENT AND ACCESS

## Objective
Remove ambiguity and get access to everything you need.

## Tasks
- [ ] Confirm whether your real target is **TensorRT** or **Tenstorrent**.
- [ ] Confirm which repo branch is the active team branch.
- [ ] Confirm where the C46 data lives on the machine you will use.
- [ ] Confirm whether you are expected to run on NVIDIA first, or CPU is acceptable for baseline sanity checks.
- [ ] Confirm who already modified the repo so you do not duplicate work.

## Output for Day 0
A short note in your own file:
- target stack = ?
- active repo/branch = ?
- dataset path = ?
- primary machine = ?
- expected first deliverable = ?

## If nobody answers quickly
Proceed with this assumption:
- baseline on regular PyTorch first,
- module experiment second,
- backend attempt third,
- document assumptions clearly.

---

# DAY 1 - BASELINE SETUP AND FIRST RUN

## Objective
Run the baseline once and remove setup uncertainty.

## Main goal
You should be able to say:
> I ran the baseline or I know exactly why it failed.

## Tasks
- [ ] Clone the active repo.
- [ ] Clone the hardware-port repo if it contains useful prior work.
- [ ] Create a clean Python environment.
- [ ] Install dependencies.
- [ ] Open the tutorial notebook or entry script.
- [ ] Verify dataset files exist.
- [ ] Run the baseline cells / script in order.
- [ ] Save the full error log if something fails.

## What to record
Create a file called:
- `notes/day1_baseline_log.md`

Record:
- machine name,
- OS,
- Python version,
- PyTorch version,
- CUDA/CPU device used,
- command used to run,
- dataset path,
- success/failure,
- exact error trace if failure.

## Exit criteria for Day 1
At least one of these must be true:
- baseline completed, or
- you identified the exact failing step.

## Cursor prompts for Day 1
Use prompts like:
- "Find the real entry point for the Kilosort-style demo in this repo."
- "List required data files for this notebook/script."
- "Explain the execution order of this notebook in simple steps."
- "Why is this import failing and what is the minimal fix?"

## If Day 1 fails badly
Do not jump to backend porting.
Instead:
- fix environment,
- fix paths,
- confirm branch,
- rerun baseline.

---

# DAY 2 - PIPELINE MAPPING

## Objective
Understand the active execution path, not just the repo structure.

## Main goal
You should be able to explain:
- what runs,
- what is defined but not used,
- which module you should target.

## Tasks
- [ ] Read the main algorithm file.
- [ ] Trace the `forward()` path or equivalent execution path.
- [ ] Write down the stages in order.
- [ ] Mark which modules are actually active.
- [ ] Identify where PCA / features are implemented.
- [ ] Identify whether PCA is active or bypassed.
- [ ] Identify one backup module if PCA is not practical.

## Recommended table to create
Create:
- `notes/day2_pipeline_map.md`

With columns:
- stage name,
- file path,
- function/class,
- active or inactive,
- likely compute-heavy?,
- likely export-friendly?,
- likely blocker.

## Your target ranking
Use this ranking unless real code inspection suggests otherwise:
1. PCA / feature conversion,
2. filtering,
3. template matching,
4. whitening,
5. clustering.

Why this order:
- PCA and filtering are easier to isolate.
- clustering is usually a poor first target because it is more logic-heavy and less friendly to simple conversion workflows.

## Exit criteria for Day 2
You must end the day with:
- one primary module chosen,
- one backup module chosen,
- one clear sentence explaining why.

## Cursor prompts for Day 2
- "Trace the active forward path in this file and summarize only what actually runs."
- "Which modules are instantiated but not actually used in the main path?"
- "Where is PCA feature conversion implemented?"
- "What is the smallest self-contained module in this pipeline for an export experiment?"

---

# DAY 3 - STANDALONE MODULE EXTRACTION

## Objective
Create a minimal, reproducible experiment for one module.

## Main goal
You should have a script that runs one module independently of the whole pipeline.

## Tasks
- [ ] Create an `experiments/` folder if it does not exist.
- [ ] Add a test script for your selected module.
- [ ] Import only the minimum required code.
- [ ] Generate synthetic input tensors with correct shapes.
- [ ] Run once on CPU.
- [ ] Run multiple times to check repeatability.
- [ ] Save output shapes and any timing.
- [ ] If available, run on CUDA as a reference.

## Suggested filenames
- `experiments/test_pca_module.py`
- `experiments/test_filter_module.py`

## What to log
Create:
- `notes/day3_module_test.md`

Record:
- module name,
- input shape,
- output shape,
- device,
- runtime,
- repeated-run consistency,
- any shape or dtype issues.

## Repeatability check
Run the module at least 3 times with the same input and note whether output changes.
This is useful because your broader research theme includes determinism.

## Exit criteria for Day 3
You must have:
- a standalone script,
- a successful local run or a precise failure point,
- a frozen test input example.

## Cursor prompts for Day 3
- "Create the smallest standalone script to run this class/function with synthetic input."
- "What input shape does this module expect?"
- "Remove all unrelated dependencies from this script."
- "Help me make this test deterministic and repeatable."

---

# DAY 4 - BACKEND ATTEMPT

## Objective
Attempt one real backend path for the isolated module.

## Important rule
This day is **not** about finishing the whole port.
It is about learning whether the module maps cleanly or not.

---

## PATH A - If your confirmed target is TensorRT

### Objective
Try the PyTorch -> ONNX -> TensorRT flow on the isolated module.

### Tasks
- [ ] Wrap the module in a minimal `torch.nn.Module` if needed.
- [ ] Export to ONNX.
- [ ] Validate the ONNX graph loads.
- [ ] Attempt TensorRT conversion / compilation.
- [ ] Save all failure messages.

### What to record
Create:
- `notes/day4_backend_attempt.md`

Record:
- export success/failure,
- ONNX ops present,
- TensorRT conversion status,
- unsupported ops,
- dynamic shape issues,
- plugins required or not.

### Good outcomes
All of these count as valid progress:
- export succeeds,
- export fails but the reason is precise,
- conversion begins but fails on unsupported ops,
- only part of the graph is supported.

### Bad outcome to avoid
Do not spend the whole day blindly changing code without logging exact blockers.

### Cursor prompts for TensorRT path
- "Wrap this function as a minimal nn.Module for ONNX export."
- "Export this module to ONNX with fixed input shape."
- "Explain this ONNX export error and suggest the smallest fix."
- "List the ops in this ONNX graph that may cause TensorRT issues."

---

## PATH B - If your confirmed target is Tenstorrent

### Objective
Check whether the isolated module can be expressed in a form suitable for the Tenstorrent software path.

### Tasks
- [ ] Identify the PyTorch ops used by the module.
- [ ] Replace unusual or object-oriented patterns with simpler functional ops if needed.
- [ ] Check what parts would require rewriting for TT-compatible execution.
- [ ] Attempt one minimal TT-style compatibility experiment if your environment allows it.
- [ ] Save exact blocker list.

### What to record
Create:
- `notes/day4_backend_attempt.md`

Record:
- ops used in the module,
- which are likely compatible,
- which need rewrite,
- shape/layout constraints,
- whether conversion or rewrite is realistic.

### Good outcomes
All of these count as valid progress:
- module is easy to rewrite,
- only some ops are problematic,
- module is not feasible without redesign but you can explain why.

### Cursor prompts for Tenstorrent path
- "List all PyTorch tensor ops used by this module."
- "Rewrite this module in simpler functional PyTorch style."
- "Which operations here are likely to require backend-specific rewriting?"
- "Minimize this module so it is easier to port to a restricted accelerator stack."

---

# DAY 5 - RESULT CONSOLIDATION

## Objective
Turn your work into something the team can use.

## Main goal
You should be able to give a short update with confidence.

## Tasks
- [ ] Clean your experiment scripts.
- [ ] Summarize the baseline status.
- [ ] Summarize the active pipeline path.
- [ ] Summarize the selected module and why.
- [ ] Summarize backend attempt results.
- [ ] List blockers clearly.
- [ ] Suggest next step.

## Deliverables to prepare
Create:
- `notes/final_week_summary.md`

Structure it as:
1. baseline status,
2. active pipeline summary,
3. chosen module,
4. experiment performed,
5. results,
6. blockers,
7. recommended next step.

## Team update template
Use this message:

> I finished the baseline check and mapped the active Kilosort-style path. For my lane, I isolated [MODULE NAME] as the first exploratory target because it is the smallest meaningful compute block for the non-AMD path. I ran a standalone experiment and attempted the [TensorRT / Tenstorrent] workflow. Current status: [SUCCESS / PARTIAL / BLOCKED]. Main blockers are: [BLOCKER 1], [BLOCKER 2], [BLOCKER 3]. My recommendation is to [NEXT STEP].

## Exit criteria for Day 5
You must have:
- one experiment folder,
- one notes folder with daily logs,
- one final summary,
- one message ready for the team.

---

# OPTIONAL DAY 6 - STRETCH WORK (ONLY IF DAY 1-5 IS DONE)

Do this only if the main plan is already complete.

## Good stretch tasks
- compare CPU vs CUDA runtime for the isolated module,
- try fixed-shape vs dynamic-shape export,
- test backup module,
- add a tiny reproducibility check,
- produce one small figure for the report.

## Do NOT do these as stretch tasks yet
- full clustering port,
- second algorithm,
- full end-to-end accelerator integration,
- major repo refactoring.

---

# DAILY CHECKLIST YOU CAN REPEAT

At the start of each day, answer these 5 questions:
1. What exact file am I working on today?
2. What exact output do I need by end of day?
3. What am I explicitly **not** doing today?
4. What will I send the team tonight?
5. What proof will show I made progress?

---

# PROJECT CONTEXT SUMMARY FOR YOU

## What the project is about
The team project is about making a Kilosort-style spike sorting pipeline more portable across hardware. The baseline is a PyTorch-style implementation in `torchbci`, and the project report frames the work around correctness and performance on the C46 dataset.

## What the team has already done
- selected `torchbci` as the code framework,
- selected C46 as the working dataset,
- wrote the project report,
- started AMD-oriented implementation work,
- identified key pipeline stages,
- agreed that acceleration and determinism matter.

## What your teammates are mainly doing
- the AMD side is the main implementation-first track,
- that side is closer to practical porting,
- your side is more exploratory and blocker-oriented.

## What your role adds
You provide the comparison point:
- AMD shows what can be ported more directly,
- your work shows what is harder and why,
- this is useful for the final report because it connects portability, system constraints, and implementation effort.

---

# FINAL RULES FOR NAZISH

## Always do these
- run baseline first,
- work on one module only,
- log exact blockers,
- save scripts and notes every day,
- keep the scope controlled.

## Never do these
- do not try to port everything,
- do not jump to a second algorithm,
- do not duplicate AMD work,
- do not mix TensorRT and Tenstorrent without confirmation,
- do not spend a full day without producing a written note.

---

# YOUR SIMPLE EXECUTION SUMMARY

## This week, you are doing exactly this:
1. confirm target,
2. run baseline,
3. map active pipeline,
4. isolate one module,
5. attempt one backend path,
6. report blockers and feasibility.

That is a complete, valid, and useful contribution.
