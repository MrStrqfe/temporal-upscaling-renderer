# Temporal Upscaling Renderer — Design Spec

## Purpose

A Vulkan renderer built around temporal upscaling and frame generation, demonstrating the core techniques behind AMD FSR and Fluid Motion Frames. Targets an AMD graphics software intern/co-op application. Runs on macOS via MoltenVK.

## Technology Stack

- **Graphics API:** Vulkan (via MoltenVK on macOS)
- **Shader Language:** Slang (compiled to SPIR-V)
- **Build System:** CMake, C++23
- **Libraries:** GLFW, volk, VMA, assimp, stb_image, GLM, Dear ImGui

## Directory Structure

```
temporal-upscaling-renderer/
├── CMakeLists.txt
├── LICENSE
├── README.md
├── libs/                          # vendored single-header libs
│   ├── stb_image.h
│   ├── volk.c
│   ├── volk.h
│   └── vmaImplementation.cpp
├── models/                        # glTF test scenes
├── shaders/                       # .slang source + compiled .spv
│   ├── forward.slang              # base forward pass (color + depth)
│   ├── velocity.slang             # motion vector generation
│   ├── taa.slang                  # temporal anti-aliasing resolve
│   ├── upscale.slang              # temporal upscaler
│   ├── sharpen.slang              # post-upscale sharpening (RCAS-style)
│   ├── frame_interp.slang         # frame interpolation
│   └── ui.slang                   # ImGui rendering
├── src/
│   ├── main.cpp
│   ├── Config.h
│   ├── App/
│   │   ├── App.h
│   │   └── App.cpp
│   ├── Window/
│   │   ├── Window.h
│   │   └── Window.cpp
│   ├── Input/
│   │   ├── Input.h
│   │   └── Input.cpp
│   ├── Renderer/                  # Vulkan setup + forward pass
│   │   ├── Renderer.h
│   │   └── Renderer.cpp
│   ├── ModelLoader/
│   │   ├── ModelLoader.h
│   │   └── ModelLoader.cpp
│   ├── Camera/                    # current + previous MVP, jitter
│   │   ├── Camera.h
│   │   └── Camera.cpp
│   ├── PostProcess/               # compute-shader post-processing chain
│   │   ├── PostProcessPipeline.h
│   │   ├── PostProcessPipeline.cpp
│   │   ├── TAAPass.h
│   │   ├── TAAPass.cpp
│   │   ├── UpscalePass.h
│   │   ├── UpscalePass.cpp
│   │   ├── SharpenPass.h
│   │   ├── SharpenPass.cpp
│   │   ├── FrameInterp.h
│   │   └── FrameInterp.cpp
│   ├── MotionVectors/
│   │   ├── MotionVectors.h
│   │   └── MotionVectors.cpp
│   └── Debug/
│       ├── DebugUI.h
│       ├── DebugUI.cpp
│       ├── Profiler.h
│       └── Profiler.cpp
└── docs/
```

## Render Pipeline Data Flow

Each frame proceeds through these stages:

### Stage 1: Input & Camera Update

- Store previous frame's view-projection matrix
- Apply sub-pixel jitter offset to projection matrix (Halton sequence, 16 samples)
- Compute new view-projection matrix

### Stage 2: Forward Pass (Graphics Pipeline)

Renders at internal resolution (`width * upscaleRatio x height * upscaleRatio`).

Outputs:
- **Color buffer** (HDR)
- **Depth buffer**
- **Velocity buffer** (per-pixel screen-space motion vectors)

Motion vectors are computed by transforming each vertex by both the current and previous MVP matrices. The fragment shader writes `currentNDC - previousNDC` to the velocity buffer.

### Stage 3: PostProcess Pipeline (Compute Shaders)

Selectable via ImGui. Four modes:

**Native** — Skip post-processing, present color directly.

**TAA** — Temporal anti-aliasing:
1. Reproject history buffer using motion vectors
2. Neighborhood clamping (3x3 color bounding box around current pixel)
3. Blend: `output = lerp(history, current, taaBlendFactor)`
4. Write output; it becomes next frame's history

**Upscale** — Temporal upscaling (TAA at lower resolution, reconstruct to native):
1. Render at 50-67% internal resolution
2. Reproject full-resolution history using motion vectors
3. Detect disocclusion via depth discontinuities and velocity divergence
4. Accumulate jittered samples into full-resolution output
5. Apply contrast-adaptive sharpening pass (RCAS-style)

**FrameInterp** — Frame interpolation (runs after TAA or Upscale):
1. Take two consecutive resolved frames and their motion vectors
2. Warp frame N forward and frame N+1 backward to midpoint
3. Blend warped frames
4. Fill disocclusion holes via depth-validity check and neighbor fallback

### Stage 4: Debug/UI Overlay

- ImGui: mode selector, split-screen comparison, motion vector visualization
- Profiler: GPU timestamp queries per pass, timing bars in overlay
- Present to swapchain

## Component Responsibilities

### App
Owns all subsystems, runs the main loop, passes delta time. Queries current PostProcess mode from DebugUI and passes it to the pipeline.

### Window / Input
GLFW window creation, surface management, keyboard/mouse input. Same pattern as reference project.

### Camera
Separated from Renderer. Stores current and previous view-projection matrices. Each frame: save previous MVP, apply jitter, compute new MVP. Exposes `getJitterOffset()`, `getCurrentVP()`, `getPreviousVP()`.

### Renderer
Vulkan setup (instance, device, allocator, swapchain) plus the forward graphics pipeline. Writes to three render targets (color, depth, velocity) instead of directly to the swapchain.

### ModelLoader
Loads glTF via assimp, extracts vertices/indices/materials. Vertices support previous-frame position computation via the previous MVP uniform in the vertex shader.

### MotionVectors
Manages the velocity render target. Vertex shader transforms position by both current and previous MVP. Fragment shader computes `currentNDC - previousNDC` and writes to velocity buffer.

### PostProcessPipeline
Orchestrates the compute chain. Owns history buffers (ping-pong pair), manages the mode enum (`Native | TAA | Upscale | FrameInterp`), calls active passes in order. Each pass is a self-contained compute dispatch with its own pipeline and descriptors.

### TAAPass
Single compute shader dispatch. Reads current color + velocity + depth + history. Reprojects, clamps to neighborhood bounds, blends, writes resolved output and updates history.

### UpscalePass
Same structure as TAA but operates across resolutions. Reads low-res color/velocity/depth, reprojects against full-res history, detects disocclusion, writes full-res output.

### SharpenPass
Lightweight compute pass after upscaling. Contrast-adaptive sharpening to counteract temporal accumulation softness.

### FrameInterp
Takes two resolved frames and motion vectors. Warps forward/backward, blends at midpoint, fills disocclusion holes.

### DebugUI
ImGui overlay: mode selector dropdown, side-by-side split view, motion vector visualization (color-coded), per-pass GPU timing bars.

### Profiler
Wraps `vkCmdWriteTimestamp` around each pass, reads back results, exposes per-pass timing data to DebugUI.

## Config

```cpp
namespace Config {
    // Window
    inline constexpr int         width             {1920};
    inline constexpr int         height            {1080};
    inline constexpr const char *title             {"Temporal Upscaling Renderer"};

    // Vulkan
    inline constexpr int         version           {1};
    inline constexpr int         maxFramesInFlight {2};
    inline constexpr int         maxBindlessTextures{256};

    // Temporal pipeline
    inline constexpr float       upscaleRatio      {0.667f};
    inline constexpr int         jitterSequenceLen {16};
    inline constexpr float       taaBlendFactor    {0.05f};
    inline constexpr float       sharpenStrength   {0.5f};
}
```

## Build Setup

CMakeLists.txt mirrors the reference project's structure:
- Lists all source files explicitly
- Vendored libs in `libs/`
- `find_package` for GLFW, VMA, assimp
- `FetchContent` for GLM and Dear ImGui
- `find_program` for `slangc` shader compiler
- Custom commands to compile `.slang` to `.spv` at build time
- `cxx_std_23`, `PROJECT_ROOT_DIR` define for runtime asset paths

## Key Differentiators from Reference Project

1. Camera as a separate module with previous-frame state for motion vectors
2. Velocity buffer generated every frame as a core render target
3. PostProcess compute pipeline with swappable modes (Native/TAA/Upscale/FrameInterp)
4. History buffer ping-pong managed by the pipeline orchestrator
5. ImGui debug panel for live comparison and GPU profiling
6. Configurable internal resolution for temporal upscaling

## Implementation Phases

1. **Base renderer** — Vulkan setup, forward pass, glTF loading, camera
2. **Motion vectors & TAA** — Velocity buffer, camera jitter, history reprojection, neighborhood clamping
3. **Temporal upscaler** — Render at reduced resolution, reconstruct to native, disocclusion detection, sharpening
4. **Frame interpolation** — Compute-based frame generation from motion vectors
5. **Debug & profiling** — ImGui panel, GPU timestamp profiling, split-screen comparison
6. **Polish** — Artifact analysis, README writeup, comparison video
