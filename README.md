![BrainChip](examples/img/BC-banner-1920x200.jpg)

# Akida Pico FPGA — Cloud Examples

Runnable examples for deploying neural-network models to **BrainChip Akida Pico** — an ultra-low-power,
event-based neural IP for always-on 1-D sensing — on the **Akida Pico FPGA cloud platform** (a single-NP
Pico IP implemented on a Xilinx FPGA and accessed through JupyterLab).

Each example takes a trained model the full path to on-device inference:

> **build → convert to a streaming (stateful) form → quantize to int8 → convert with Akida's MetaTF
> toolchain → map onto the Pico → measure and run on hardware.**

The Akida software toolchain — **MetaTF** (`akida`, `cnn2snn`, `quantizeml`, `akida_models`) — and full
API documentation is here: **https://doc.brainchipinc.com/index.html**

## Examples

| Example | Task | What it demonstrates |
|---------|------|----------------------|
| [Keyword Spotting](examples/kws/) | 12-class speech-command recognition (audio, 16 kHz) | Streaming SSM keyword spotting on Pico — data → stateful conversion → quantization → Akida mapping → latency/throughput/power → streaming inference. |
| [Bearing Fault Detection](examples/fault_detection/) | Multi-label vibration fault detection (accelerometer, 42 kHz) | The same pipeline on a 1-D vibration stream — real-time multi-label fault detection, hardware metrics, a float-vs-Akida comparison, and an accuracy-vs-decision-latency study. |
| [sEMG Gesture Recognition](examples/semg_gesture/) | 49-gesture hand-gesture recognition from forearm sEMG (12 channels, 2 kHz), NinaPro DB2 | A self-contained tutorial on continuous streaming recognition: dataset and protocol, stateful conversion, int8 quantization, the real Pico mapping constraints, a wake trigger, and results against the published DB2 literature. This model does not fit the shipped Pico FPGA, so it reports no hardware performance. |

Each example folder has its own `README.md` with the details.

## Setup

The Akida Cloud host already has the Pico FPGA attached and conda/Python available.

1. **Fetch the large example assets** (needed by the sEMG example, whose weights, calibration
   streams and held-out segments are stored with Git LFS):

   ```bash
   git lfs install
   git lfs pull
   ```

2. **Install dependencies:**

   ```bash
   conda install -c conda-forge jupyterlab ffmpeg
   pip install -r requirements.txt
   ```

   (`ffmpeg` is used by `tensorflow_datasets` to prepare the Speech Commands dataset for the Keyword
   Spotting example.)

3. **Launch JupyterLab:**

   ```bash
   ./start-jupyterlab.sh
   ```

4. **Open an example**, for example `examples/kws/kws_sc12.ipynb`,
   `examples/fault_detection/fault_detection_inference.ipynb` or
   `examples/semg_gesture/ninapro_db2_tenns_r.ipynb`, and run all cells in order.

## The Pico device

The Keyword Spotting and Bearing Fault Detection examples run on a real Akida Pico device. Confirm
it is visible before running:

```bash
akida devices            # or:  python -c "import akida; print(akida.devices())"
```

You should see one device. Hardware and platform details are in
[examples/Akida_Cloud_Specs.md](examples/Akida_Cloud_Specs.md).

The sEMG Gesture Recognition example needs no device: it checks the model against the Pico
constraints using `akida.PicoIP()`, a virtual device carrying the board's memory configuration.
