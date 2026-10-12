# Image Grayscaling: Serial vs OpenMP vs CUDA

A High Performance Computing (HPC) mini-project that converts an RGB image to grayscale three ways and benchmarks them against each other: a **sequential CPU** implementation, a **multithreaded OpenMP** implementation, and a **CUDA GPU** kernel. The whole pipeline runs from a single Google Colab notebook and ends with a visual performance report.

## Highlights

- Same grayscale algorithm in three execution models, so the comparison is apples to apples
- Image is uploaded in Colab, embedded into generated CUDA/C source as a byte array, compiled with `nvcc`, and run
- Each version is timed over 10 runs and averaged
- Outputs of all three versions are read back into Python and plotted side by side with the original
- Auto-generated report (`grayscale_report.png`) with execution time, speedup, and estimated cycle counts

## How it works

Grayscale conversion uses the standard luminance weights:

```
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

| Version | Approach |
|---|---|
| Sequential CPU | Single loop over every pixel |
| OpenMP CPU | `#pragma omp parallel for` with static scheduling across all available cores |
| CUDA GPU | One thread per pixel, 256 threads per block, `ceil(N / 256)` blocks |

Pipeline:

1. Install dependencies and upload an image (OpenCV decodes it to RGB)
2. Python generates `grayscale.cu`, embedding the pixel data and image dimensions as compile-time constants
3. `nvcc -O2 -Xcompiler -fopenmp` compiles and the binary runs the three benchmarks
4. Python parses timings and the three grayscale outputs from stdout
5. Matplotlib builds the report: original and grayscale images, time bar chart, speedup bar chart, metrics panel

## Requirements

- Google Colab with a **GPU runtime** (Runtime → Change runtime type → GPU)
- CUDA toolkit with `nvcc` (preinstalled on Colab GPU runtimes)
- Python packages (installed by the first cell): `numpy`, `matplotlib`, `opencv-python-headless`

## Usage

1. Open `hpc_jackfruit_final.ipynb` in Google Colab
2. Switch to a GPU runtime
3. Run the cells in order
4. When prompted, upload an image (JPG/PNG)
5. View the report inline; it is also saved to `/content/grayscale_report.png`

## Sample results

Test image: `lily_flower.jpg`, 640 × 960 (614,400 pixels), Colab runtime with 2 CPU threads.

| Method | Time (avg of 10 runs) | Speedup |
|---|---|---|
| Sequential CPU | 2.3026 ms | 1.00× |
| OpenMP CPU (2 threads) | 1.9192 ms | 1.20× |
| CUDA GPU | 0.013840 ms | 166.37× |

## Notes and limitations

- **GPU timing is kernel-only.** It uses CUDA events around the kernel launches and does not include host-to-device or device-to-host memory transfers. End-to-end, including transfers, the GPU speedup would be considerably smaller.
- **Cycle counts are estimates**, derived from time assuming 2 GHz for the CPU and 1 GHz for the GPU. They are not hardware counter measurements.
- OpenMP speedup is limited by the thread count of the Colab instance (2 threads here), and the operation is memory-bound.
- Pixel data is embedded in the source as hex, so very large images will make the generated `.cu` file big and slow to compile.
- Results are printed through stdout and parsed back with regex, which is simple but not suited to large-scale use.

## Possible extensions

- Include transfer time in the GPU measurement and report both kernel-only and end-to-end speedups
- Load the image at runtime instead of embedding it, and test multiple image sizes to show scaling
- Verify output correctness automatically (compare the three outputs pixel by pixel)
- Try pinned memory, streams, or vectorized loads (e.g. `uchar4`) in the CUDA kernel

## Project structure

```
.
├── hpc_jackfruit_final.ipynb   # Full pipeline: upload, codegen, compile, benchmark, plot
└── README.md
```

Generated at runtime in Colab: `grayscale.cu`, `grayscale` (binary), `grayscale_report.png`.
