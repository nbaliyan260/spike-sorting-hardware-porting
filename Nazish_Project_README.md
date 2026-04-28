The plan in that file is based on your project report, which says the project is to port a PyTorch-based Kilosort-style pipeline in torchbci to AMD and Tenstorrent-class hardware, using C46 as the evaluation dataset and focusing on correctness and performance.

Your main lane is not AMD. Based on your team split, Nazish is on the exploratory non-AMD lane. In practice, that means: run the baseline first, understand the active pipeline, pick one module, try the conversion/porting path for that module, and document blockers clearly. The README I wrote is built around that role.

One important correction is that your chats mix up TensorRT and Tenstorrent, but they are not the same target. TensorRT is NVIDIA’s inference SDK; its standard workflows are ONNX conversion or Torch-TensorRT, and unsupported operators need plugins. Tenstorrent uses a different stack: TT-NN recommends rewriting the torch model into functional torch APIs and then switching ops to ttnn, while TT-XLA ingests PyTorch through torch-xla. So the first thing you should do before coding is confirm with the team whether your real target is TensorRT or actual Tenstorrent hardware/software.

I also checked the public team codebase I could access. The two repo links you pasted were not publicly fetchable from here, so I used the public fork that surfaced from your team’s GitHub activity. That repo has demo/tutorial_kilosort4.ipynb as the starting notebook, torchbci/algorithms/kilosort.py as the Kilosort-style pipeline, torchbci/block/* for the modular stages, compat.py for CPU/CUDA/ROCm backend handling, and a repo README that still lists “Get Kilosort work within this framework” as a to-do—so it should be treated as an in-progress baseline, not a finished production pipeline.

The most important code-level finding is this: the public fork already defines a PCA block, Kilosort4PCFeatureConversion, but the current Kilosort4Algorithm.forward() path instantiates it and then bypasses it, assigning spike_pc_features = spike_features before clustering. So if you work on PCA first, treat it as a standalone module experiment, not an already integrated end-to-end stage. That is why the README tells you to run the baseline, map the active path, and then isolate one module cleanly before attempting TensorRT or Tenstorrent conversion.

I also grounded the README in the Kilosort4 paper itself: Kilosort4 is Python + PyTorch, its preprocessing pipeline includes CAR, temporal filtering, whitening, and drift correction, and its feature pipeline combines template deconvolution, PCA-style feature extraction, and graph-based clustering. That matters because it tells you which parts are structurally more promising for modular porting and which parts are more entangled.














# Nazish README - Spike Sorting Hardware Porting Plan

## 1) What this project is about

This project is about taking a Kilosort-style spike sorting pipeline and making it portable beyond the original NVIDIA/CUDA-oriented workflow. The team project report defines the goal as porting a PyTorch-based Kilosort-style pipeline in `torchbci` to AMD and Tenstorrent-class hardware, and evaluating correctness and performance on the C46 dataset.

## 2) What has already been done

Based on the report and your team chat:
- The team has already chosen `torchbci` + C46 as the baseline setup.
- The team has already written a short report and identified the main pipeline stages: preprocessing, spike detection, feature extraction, and clustering.
- The AMD side has been the primary implementation track.
- Your lane is the exploratory Tenstorrent / TensorRT lane, not the AMD lane.

## 3) Important clarification before coding

Your chat mixes **Tenstorrent** and **TensorRT**, but they are not the same thing.
- **TensorRT** is NVIDIA’s inference SDK for NVIDIA GPUs.
- **Tenstorrent** is different hardware with its own software stack (TT-NN / TT-XLA / TT-Metalium).

So before you spend serious time coding, send one message to the team:

> “Please confirm my target for this week: are we evaluating NVIDIA TensorRT, or actual Tenstorrent hardware/software (TT-NN / TT-XLA)? I will proceed on one track only.”

If the answer is **Tenstorrent**, do not build a TensorRT-only solution.
If the answer is **TensorRT**, do not spend time on TT-NN / TT-XLA.

## 4) What your role is

Your job is **not** to port the full project end-to-end by yourself.
Your job is to:
1. run and understand the baseline,
2. identify a self-contained compute-heavy module,
3. try the conversion / porting path for that module,
4. document exactly what works and what breaks,
5. report feasibility and blockers.

In one sentence:

> I own the exploratory non-AMD lane: baseline understanding + one-module conversion attempt + blocker analysis.

## 5) What repo structure matters for you

In the public hardware-port fork, the useful files are:
- `demo/tutorial_kilosort4.ipynb` - baseline notebook
- `torchbci/algorithms/kilosort.py` - main Kilosort-style pipeline
- `torchbci/compat.py` - device/backend helper
- `torchbci/block/filter.py`
- `torchbci/block/detection.py`
- `torchbci/block/alignment.py`
- `torchbci/block/templatematching.py`
- `torchbci/block/clustering.py`

The tutorial notebook uses:
- `../data/c46_npx_raw.bin`
- `../data/c46_all_spikes.mat`
- `../data/wTEMP.npz`

## 6) Very important code insight

In the current public Kilosort-style implementation, the pipeline creates a PCA module (`Kilosort4PCFeatureConversion`), but the forward path currently bypasses it and uses raw spike features directly for clustering.

That means:
- PCA is implemented as a separate module,
- but PCA is **not currently active** in the public forward pipeline,
- so if you work on PCA conversion, treat it as a **module-level experiment**, not as an already integrated end-to-end stage.

This is important because earlier discussions focused on PCA as the likely optimization target, but the current code path still needs integration work.

## 7) What you should do now

### Phase A - Confirm target stack (first task, 10 minutes)
Send one message and get a direct answer:
- “Am I targeting actual Tenstorrent or NVIDIA TensorRT?”

Then follow one path only.

---

### Phase B - Run the baseline (mandatory)
Goal: prove you can run the Kilosort-style baseline before attempting any port.

Steps:
1. Clone the active repo your team is actually using.
2. Create an environment and install dependencies.
3. Open `demo/tutorial_kilosort4.ipynb`.
4. Verify the C46-related files exist.
5. Run the notebook cells in order.
6. Confirm you can instantiate `Kilosort4Algorithm` and call `forward(data_tensor)`.

What you must record:
- environment details,
- which device ran the code,
- whether it completed,
- any errors,
- the exact input files used.

Exit condition:
- baseline notebook runs, or
- you have a clean error log showing exactly where it fails.

---

### Phase C - Read the pipeline and map the active path
Goal: understand which operators are actually used in the current implementation.

From `torchbci/algorithms/kilosort.py`, trace the forward path:
1. CAR
2. high-pass filtering
3. whitening
4. detection
5. clustering

Also inspect:
- detection internals,
- template matching,
- any use of `torchaudio`, `torch.linalg`, `torch.cov`, `torch.pca_lowrank`, or custom ops.

Make a small table for yourself:
- module name,
- file,
- active or inactive,
- likely portability difficulty,
- likely backend blocker.

Exit condition:
- you have a one-page pipeline map.

---

### Phase D - Pick your module
Pick **one** module only.

Recommended priority:
1. **If your target is Tenstorrent:** start with a small, self-contained module such as PCA feature conversion or filtering, because full detection/clustering will be much harder.
2. **If your target is TensorRT:** start with PCA feature conversion first, then filtering if PCA export fails immediately.

My recommendation for you:
- **Primary experiment:** `Kilosort4PCFeatureConversion`
- **Backup experiment:** `Kilosort4Filtering`

Why:
- PCA is self-contained and easier to isolate.
- Filtering is active in the pipeline and easier to benchmark.
- Whitening and clustering are more likely to introduce unsupported or more complex operator chains early.

Exit condition:
- one module selected,
- one backup module selected.

---

### Phase E - Isolate the module
Create a standalone test file, not a full-pipeline port.

Suggested files:
- `experiments/test_pca_module.py`
- `experiments/test_filter_module.py`

For each test file:
1. import only the selected class,
2. create synthetic input tensors with the right shape,
3. run the module on CPU,
4. run the module on CUDA if available,
5. save input/output shapes,
6. save timing,
7. save any exception trace.

You are trying to answer:
- does this module run independently,
- what operators does it use,
- is it deterministic enough to compare repeated runs,
- is it portable to the chosen backend path.

Exit condition:
- you have a standalone reproducible test for one module.

---

### Phase F - Backend-specific implementation

#### If your confirmed target is TensorRT
Use this workflow:
1. wrap the module in a small `torch.nn.Module`,
2. export to ONNX,
3. inspect export success/failure,
4. try conversion with TensorRT tooling,
5. record unsupported ops.

What success means:
- export works and conversion starts,
- or export fails in a way you can clearly explain.

What failure still counts as good progress:
- unsupported ops,
- dynamic shape issues,
- operator mismatch,
- custom plugin requirement,
- partial graph conversion only.

#### If your confirmed target is Tenstorrent
Use this workflow:
1. start from a tiny PyTorch module,
2. try the official Tenstorrent path,
3. first test whether the op can be rewritten in functional PyTorch,
4. then try TT-NN or TT-XLA-compatible conversion,
5. record what would need rewriting.

What success means:
- the module can be expressed in TT-compatible ops,
- or you identify exactly which PyTorch ops block the path.

What failure still counts as good progress:
- unsupported operation coverage,
- shape/layout issues,
- explicit rewrite requirements,
- hardware-specific constraints.

Exit condition:
- you have one real backend experiment completed.

---

### Phase G - Evaluate and document
You own the documentation of the exploratory path.

Create a short note with five sections:
1. module chosen,
2. why chosen,
3. test setup,
4. results,
5. blockers / next steps.

Also record:
- runtime,
- repeated-run consistency,
- operator list,
- export/conversion status,
- whether fallback to PyTorch was needed.

## 8) What you should NOT do

Do not:
- work on AMD implementation in parallel,
- try to port the whole Kilosort pipeline alone,
- start with clustering,
- spend days debugging infrastructure before confirming the target stack,
- assume TensorRT and Tenstorrent are interchangeable.

## 9) Your deliverables

You should finish with these deliverables:
1. baseline run log,
2. pipeline map,
3. one standalone module test,
4. one backend conversion/porting attempt,
5. blocker report,
6. a short update for the team.

## 10) The exact update you should send to the team after the first real work block

Use this template:

> I ran the baseline Kilosort-style notebook and mapped the active pipeline stages. In the current public fork, PCA exists as a module but appears to be bypassed in the forward path, so I am treating it as a standalone experiment rather than an already integrated stage. I will target one module only for the exploratory non-AMD lane: first PCA feature conversion, with filtering as backup. I am now proceeding on the confirmed target stack: [TensorRT or Tenstorrent]. Next update will include export / conversion status and blockers.

## 11) What “success” looks like for you this week

Success is **not** full port completion.
Success is:
- you understand the pipeline,
- you ran the baseline,
- you isolated one module,
- you attempted one backend path,
- you can clearly explain the blockers.

That is a valid and useful project contribution.
