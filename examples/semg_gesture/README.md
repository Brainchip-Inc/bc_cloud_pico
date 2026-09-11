# Continuous sEMG Gesture Recognition with TENNs-R (NinaPro DB2)

This example is a **self-contained tutorial notebook**. It takes a real always-on sensing problem,
recognising hand gestures from 12 channels of forearm surface EMG, and walks the whole path from the
public **NinaPro DB2** dataset to a streaming int8 model checked against **Akida Pico**.

The model is **TENNs-R**, a state space model that trains as a convolution and runs as a recurrence.
At **96,128 parameters** and int8 it reaches **85.31 % plus or minus 4.37** on the 49 gesture classes
averaged over all 40 DB2 subjects, scored on a continuous stream rather than on pre-cut windows.

> **Read this before you look for performance numbers.** Unlike the other examples in this
> repository, **this model does not fit the shipped Akida Pico FPGA**, so the notebook reports **no
> frames per second, no latency and no power**. Section 10 runs the real mapper, shows the refusal,
> explains each constraint and states the Pico IP configuration that accommodates the model. Every
> accuracy figure in the notebook is produced on the CPU with the same MetaTF toolchain, end to end.

The notebook is self-contained by design: there is no `*_utils.py` sidecar. Every function, the model
builder and all plotting code live in the notebook itself, so it can be read, run and modified in one
place.

## Folder structure

```
ninapro_db2/
|- ninapro_db2_tenns_r.ipynb              # The tutorial notebook, the only code in this example
|- README.md
|- weights/
|   |- tenns_r_db2_s23.weights.h5         # Trained float weights, subject 23        (1.3 MB, LFS)
|- calibration/
|   |- db2_s23_calibration.npz            # 98 x 5 s int16 PTQ streams               (21 MB, LFS)
|- test_data/
|   |- db2_s23_test_segments.npz          # 98 held-out passes with per-step labels  (36 MB, LFS)
|   |- db2_s23_protocol_excerpt.npz       # 107 s of contiguous recording for the protocol figure
|                                         #                                          (4.8 MB, LFS)
|- results/
    |- exp12_cohort.csv                   # Float and int8 accuracy for all 40 subjects
    |- literature_db2.csv                 # Published DB2 results used in the comparison figure
```

| Path | Purpose |
|------|---------|
| `ninapro_db2_tenns_r.ipynb` | dataset -> model -> stateful conversion -> int8 quantization -> Akida conversion and Pico budget -> streaming inference -> trigger -> 40-subject results |
| `weights/tenns_r_db2_s23.weights.h5` | The trained float weights for subject 23, loaded into the model the notebook builds |
| `calibration/db2_s23_calibration.npz` | Representative streams for post-training quantization, drawn from training repetitions only |
| `test_data/db2_s23_test_segments.npz` | The subject's complete held-out set: repetitions 2 and 5 of all 49 gestures |
| `test_data/db2_s23_protocol_excerpt.npz` | One contiguous stretch of recording, used to draw the training and scoring protocol |
| `results/*.csv` | The cohort and literature tables behind the two closing figures |

## Quick start

1. **Check out the binary assets.** They are stored with **Git LFS**. If `git lfs` was installed
   before you cloned, they are already in place. Otherwise:

   ```bash
   git lfs install          # once per machine
   git lfs pull             # fetch the real files for this repository
   ```

   A quick way to tell whether you have the real files: `test_data/db2_s23_test_segments.npz` should
   be about 36 MB. If it is a few hundred bytes of text beginning `version https://git-lfs...`, LFS
   has not fetched it yet and the notebook will fail to load it.

2. **Prerequisites.** Python packages: `akida`, `cnn2snn`, `quantizeml`, `akida_models`,
   `tensorflow`, `tf_keras`, `numpy`, `matplotlib`. These come from the repository's
   `requirements.txt`. **No dataset download is needed**, everything the notebook reads is in this
   folder.

3. **Pico.** No Pico device is required. The Pico section maps against `akida.PicoIP()`, a virtual
   device that carries the real board's memory configuration, so the check runs anywhere.

4. **Run.** Open `ninapro_db2_tenns_r.ipynb` and run all cells in order (or **Run All**). Expect
   **four to ten minutes** end to end on a CPU, depending on the host: the two quantization passes
   and the two streaming passes over all 98 held-out segments dominate.

## What the notebook should print

| Step | Expected |
|---|---|
| Model built | 96,128 trainable parameters |
| Converted to stateful | input `(1, 16, 12)`, output `(1, 1, 50)`, 93,616 parameters |
| Conv mode vs streaming | argmax agreement 100 %, relative difference about 1e-5 |
| Streaming, subject 23, float | `acc_49` 86.85 % |
| Streaming, subject 23, int8 | `acc_49` 86.82 % |
| Mapping on `akida.PicoIP()` | refused: input row not a power of two, FSRAM and TSRAM over budget |
| Mapping on a Pico IP with 2x FSRAM and 2x TSRAM | mapped, one hardware sequence |
| Cohort, 40 subjects | float 85.39 %, int8 85.31 % |

Subject 23's two accuracies reproduce that subject's recorded training-run results exactly, which is
the check that the shipped assets carry the same preparation the model was trained with.

## Where the shipped data came from

Everything here derives from the public NinaPro DB2 release for **subject 23**,
`https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s23.zip` (463 MiB), plus that subject's trained
run. Please read NinaPro's own terms of use before redistributing any part of the dataset, and cite
the dataset paper:

> M. Atzori et al., *Electromyography data for non-invasive naturally-controlled robotic hand
> prostheses*, Scientific Data 1:140053, 2014. DOI 10.1038/sdata.2014.53.

| File | How it was produced |
|---|---|
| `weights/tenns_r_db2_s23.weights.h5` | The trained float checkpoint of subject 23's run, unmodified |
| `calibration/db2_s23_calibration.npz` | One 5 s stream per gesture per calibration repetition, starting 1 s before each movement onset, drawn from **training repetitions 1 and 4 only**, scaled to int16 with the subject's per-channel training scale |
| `test_data/db2_s23_test_segments.npz` | Every held-out repetition (2 and 5) of all 49 gestures, cut with the same segment geometry the evaluation uses: half the preceding rest, the movement, half the following rest, chunk-aligned, with the per-step label and time-since-onset columns |
| `test_data/db2_s23_protocol_excerpt.npz` | A contiguous stretch of exercise 1 covering all six repetitions of the first two gestures, with both label tracks, used only for the protocol figure |
| `results/exp12_cohort.csv` | Per-subject float and int8 accuracy over the 49 gesture classes, for all 40 subjects, from the same experiment |
| `results/literature_db2.csv` | Published DB2 results transcribed from the cited papers, with their class counts, window lengths and parameter counts |

The int16 conversion uses the per-channel scale fitted on that subject's **training** repetitions,
which is why the notebook's accuracy matches the recorded result to the last digit. Per-channel
amplitude spans three orders of magnitude within one subject, so a single global scale would not
work.

## Pipeline overview

1. **The use case and the dataset**: what continuous sEMG looks like and why a stream is harder than
   a window.
2. **Forty subjects, forty models**: why sEMG work is subject-dependent, and what that assumes.
3. **The protocol**: how repetition segments tile the recording, which repetitions train, and how
   each held-out repetition is scored as one independent pass from a reset state.
4. **Why a state space model**: buffer, re-reads and context, and the fact that Pico has no attention
   primitive at all.
5. **Build** the model on `akida_models.layer_blocks.kernelized_block`, 96,128 parameters.
6. **Convert to stateful** first, then **quantize to int8**. That order matters and the notebook
   explains why.
7. **Convert with `cnn2snn`** and **map against Pico**, including the padded 16-channel deploy width
   and the four memory budgets.
8. **Stream** every held-out pass through the int8 model, 8 ms at a time.
9. **Build a wake trigger** from the model's own rest probability and measure detection latency and
   duty cycle.
10. **Results**: all 40 subjects, then accuracy against parameter count for the published DB2
    literature, with an explicit note on what is and is not comparable.

Run the cells in order; later sections depend on earlier ones.
