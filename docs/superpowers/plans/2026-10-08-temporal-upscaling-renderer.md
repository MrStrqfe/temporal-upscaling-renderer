# Temporal Upscaling Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Vulkan renderer with temporal upscaling and frame generation, demonstrating the core techniques behind AMD FSR and Fluid Motion Frames.

**Architecture:** Forward renderer outputs color, depth, and velocity buffers. A compute-shader post-processing chain (TAA, temporal upscaling, frame interpolation) consumes these buffers using ping-ponged history. An ImGui debug panel toggles modes and displays GPU timings.

**Tech Stack:** Vulkan 1.3 (MoltenVK), Slang shaders, C++23, CMake, GLFW, volk, VMA, assimp, stb_image, GLM, Dear ImGui

**Spec:** `docs/superpowers/specs/2026-10-08-temporal-upscaling-renderer-design.md`

## Global Constraints

- C++23 (`cxx_std_23`)
- Vulkan 1.3 (MoltenVK on macOS does not support 1.4 — no `hostImageCopy`, no `VK_KHR_UNIFIED_IMAGE_LAYOUTS`)
- CMake minimum 3.24 (for `FetchContent` improvements)
- Shaders written in Slang, compiled to SPIR-V via `slangc`
- All Vulkan functions loaded via volk (no linking to Vulkan loader)
- GPU memory managed via VMA with volk function import
- Dynamic rendering (`VK_KHR_dynamic_rendering`) — no render pass objects
- Synchronization2 (`VK_KHR_synchronization2`) — pipeline barrier style matches reference project
- Buffer device address for vertex/material data (push constants, no vertex input bindings)
- `PROJECT_ROOT_DIR` CMake define for runtime asset paths
- Source files listed explicitly in CMakeLists.txt (no globbing)
- Follow reference project's code style: designated initializers, `std::views`, `m_` prefix for members, `checkResult()` pattern, brace initialization

---

### Task 1: CMake Build System & Vendored Libraries

**Files:**
- Create: `CMakeLists.txt`
- Create: `libs/volk.c`
- Create: `libs/volk.h`
- Create: `libs/vmaImplementation.cpp`
- Create: `libs/stb_image.h`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: A build system that compiles an empty `main.cpp` and links all dependencies. Later tasks add source files to the `SOURCE` list.

- [ ] **Step 1: Download vendored headers**

Download volk, stb_image headers, and create VMA implementation file:

```bash
# volk (single-file Vulkan loader)
curl -L -o libs/volk.h "https://raw.githubusercontent.com/zeux/volk/master/volk.h"
curl -L -o libs/volk.c "https://raw.githubusercontent.com/zeux/volk/master/volk.c"

# stb_image
curl -L -o libs/stb_image.h "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
```

Create `libs/vmaImplementation.cpp`:
```cpp
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
```

- [ ] **Step 2: Write CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.24)

project(temporal-upscaling-renderer)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# packages
find_package(glfw3 3.3 REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(assimp REQUIRED)

# FetchContent dependencies
include(FetchContent)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.8
)
FetchContent_MakeAvailable(imgui)

# ImGui source files (core + Vulkan + GLFW backends)
set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

# shader compilation
find_program(SLANGC_PATH NAMES slangc)
if (NOT SLANGC_PATH)
    message(WARNING "slangc not found — shaders must be compiled manually")
endif()

set(SHADER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

function(compile_slang_shader SHADER_NAME ENTRY_POINT STAGE)
    if (SLANGC_PATH)
        add_custom_command(
            OUTPUT ${SHADER_OUTPUT_DIR}/${SHADER_NAME}_${STAGE}.spv
            COMMAND ${SLANGC_PATH}
                ${SHADER_SOURCE_DIR}/${SHADER_NAME}.slang
                -entry ${ENTRY_POINT}
                -stage ${STAGE}
                -target spirv
                -o ${SHADER_OUTPUT_DIR}/${SHADER_NAME}_${STAGE}.spv
            DEPENDS ${SHADER_SOURCE_DIR}/${SHADER_NAME}.slang
            COMMENT "Compiling ${SHADER_NAME}.slang (${STAGE})"
        )
    endif()
endfunction()

# Forward pass shaders
compile_slang_shader(forward vertexMain vertex)
compile_slang_shader(forward fragmentMain fragment)

# Compile all shaders
add_custom_target(compile_shaders DEPENDS
    ${SHADER_OUTPUT_DIR}/forward_vertex.spv
    ${SHADER_OUTPUT_DIR}/forward_fragment.spv
)

# source files
set(SOURCE
    src/main.cpp
    src/App/App.cpp
    src/Window/Window.cpp
    src/Input/Input.cpp
    src/Renderer/Renderer.cpp
    src/ModelLoader/ModelLoader.cpp
    src/Camera/Camera.cpp
)

set(LIBS
    libs/volk.c
    libs/vmaImplementation.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE} ${LIBS} ${IMGUI_SOURCES})
add_dependencies(${PROJECT_NAME} compile_shaders)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)

target_include_directories(${PROJECT_NAME} PRIVATE
    src
    libs
    models
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    GLFW_INCLUDE_VULKAN
    STB_IMAGE_IMPLEMENTATION
    VMA_STATIC_VULKAN_FUNCTIONS=0
    VMA_DYNAMIC_VULKAN_FUNCTIONS=0
    PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}/"
    SHADER_DIR="${SHADER_OUTPUT_DIR}/"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    GPUOpen::VulkanMemoryAllocator
    assimp::assimp
    glm::glm
)
```

- [ ] **Step 3: Update .gitignore**

Add build directory and IDE files:

```gitignore
# Build
build/
out/

# IDE
.idea/
.vscode/
*.swp

# Compiled shaders
*.spv

# OS
.DS_Store
```

- [ ] **Step 4: Create minimal main.cpp**

Create `src/main.cpp`:
```cpp
#include <print>

int main() {
    std::println("Temporal Upscaling Renderer");
    return 0;
}
```

- [ ] **Step 5: Verify build**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/temporal-upscaling-renderer
```

Expected: prints "Temporal Upscaling Renderer" and exits.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt libs/ src/main.cpp .gitignore
git commit -m "feat: add CMake build system with vendored libs and dependency setup"
```

---

### Task 2: Window, Input & Config

**Files:**
- Create: `src/Config.h`
- Create: `src/Window/Window.h`
- Create: `src/Window/Window.cpp`
- Create: `src/Input/Input.h`
- Create: `src/Input/Input.cpp`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Config::width`, `Config::height`, `Config::title`, `Config::version`, `Config::maxFramesInFlight`, `Config::maxBindlessTextures`, `Config::upscaleRatio`, `Config::jitterSequenceLen`, `Config::taaBlendFactor`, `Config::sharpenStrength`
  - `Window::initWindow()`, `Window::shouldClose()`, `Window::isKeyPressed(int)`, `Window::closeWindow()`, `Window::pollEvents()`, `Window::getTime()`, `Window::captureCursor()`, `Window::getRequiredInstanceExtensions()`, `Window::createWindowSurface(VkInstance, const VkAllocationCallbacks*, VkSurfaceKHR*)`, `Window::getFramebufferSize()`, `Window::getWindow()`, `Window::cleanup()`
  - `InstanceExtensionInfo{extensionCount, extensions}`, `FramebufferSize{width, height}`
  - `Input::processUserInput(Window&)`

- [ ] **Step 1: Create Config.h**

Create `src/Config.h`:
```cpp
#pragma once

namespace Config {
    // Window
    inline constexpr int         width              {1920};
    inline constexpr int         height             {1080};
    inline constexpr const char *title              {"Temporal Upscaling Renderer"};

    // Vulkan
    inline constexpr int         version            {1};
    inline constexpr int         maxFramesInFlight  {2};
    inline constexpr int         maxBindlessTextures{256};

    // Temporal pipeline
    inline constexpr float       upscaleRatio       {0.667f};
    inline constexpr int         jitterSequenceLen  {16};
    inline constexpr float       taaBlendFactor     {0.05f};
    inline constexpr float       sharpenStrength    {0.5f};
}
```

- [ ] **Step 2: Create Window.h**

Create `src/Window/Window.h`:
```cpp
#pragma once
#include "volk.h"
#include <GLFW/glfw3.h>
#include <cstdint>

struct InstanceExtensionInfo {
    uint32_t     extensionCount{};
    const char **extensions    {};
};

struct FramebufferSize {
    int width{}, height{};
};

class Window {
public:
    void initWindow();
    bool shouldClose() const;
    bool isKeyPressed(int key) const;
    void closeWindow();
    void pollEvents();
    float getTime();
    void captureCursor();
    GLFWwindow *getWindow() const;
    InstanceExtensionInfo getRequiredInstanceExtensions() const;
    void createWindowSurface(VkInstance instance, const VkAllocationCallbacks *allocator, VkSurfaceKHR *surface);
    FramebufferSize getFramebufferSize();
    void cleanup();
private:
    GLFWwindow *m_window{};
};
```

- [ ] **Step 3: Create Window.cpp**

Create `src/Window/Window.cpp`:
```cpp
#include "Window.h"
#include "Config.h"
#include <stdexcept>

void Window::initWindow() {
    if (glfwInit() == GLFW_FALSE) {
        throw std::runtime_error("Error: Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(Config::width, Config::height, Config::title, nullptr, nullptr);

    if (m_window == nullptr) {
        throw std::runtime_error("Error: Failed to create GLFW window");
    }
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(m_window);
}

bool Window::isKeyPressed(int key) const {
    return glfwGetKey(m_window, key) == GLFW_PRESS;
}

void Window::closeWindow() {
    glfwSetWindowShouldClose(m_window, GLFW_TRUE);
}

void Window::pollEvents() {
    glfwPollEvents();
}

float Window::getTime() {
    return static_cast<float>(glfwGetTime());
}

void Window::captureCursor() {
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

GLFWwindow *Window::getWindow() const {
    return m_window;
}

InstanceExtensionInfo Window::getRequiredInstanceExtensions() const {
    uint32_t glfwExtensionCount{};
    const char **glfwExtensions{glfwGetRequiredInstanceExtensions(&glfwExtensionCount)};
    return {glfwExtensionCount, glfwExtensions};
}

void Window::createWindowSurface(VkInstance instance, const VkAllocationCallbacks *allocator, VkSurfaceKHR *surface) {
    if (glfwCreateWindowSurface(instance, m_window, allocator, surface) != VK_SUCCESS) {
        throw std::runtime_error("Error: Failed to create Vulkan window surface");
    }
}

FramebufferSize Window::getFramebufferSize() {
    int width{}, height{};
    glfwGetFramebufferSize(m_window, &width, &height);
    return {width, height};
}

void Window::cleanup() {
    glfwDestroyWindow(m_window);
    glfwTerminate();
}
```

- [ ] **Step 4: Create Input.h and Input.cpp**

Create `src/Input/Input.h`:
```cpp
#pragma once
class Window;

class Input {
public:
    void processUserInput(Window &window) const;
};
```

Create `src/Input/Input.cpp`:
```cpp
#include "Input.h"
#include "Window/Window.h"
#include <GLFW/glfw3.h>

void Input::processUserInput(Window &window) const {
    if (window.isKeyPressed(GLFW_KEY_ESCAPE)) {
        window.closeWindow();
    }
}
```

- [ ] **Step 5: Verify — open a window**

Update `src/main.cpp`:
```cpp
#include "Window/Window.h"
#include "Input/Input.h"
#include <iostream>
#include <exception>

int main() {
    try {
        Window window{};
        Input input{};
        window.initWindow();

        while (!window.shouldClose()) {
            window.pollEvents();
            input.processUserInput(window);
        }

        window.cleanup();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}
```

Build and run. Expected: a 1920x1080 black window opens, pressing Escape closes it.

- [ ] **Step 6: Commit**

```bash
git add src/Config.h src/Window/ src/Input/ src/main.cpp
git commit -m "feat: add Window, Input, and Config modules"
```

---

### Task 3: Camera Module

**Files:**
- Create: `src/Camera/Camera.h`
- Create: `src/Camera/Camera.cpp`

**Interfaces:**
- Consumes: `Config::jitterSequenceLen`, `Config::width`, `Config::height`, `Config::upscaleRatio`
- Produces:
  - `Camera::update(float deltaTime)` — saves previous VP, computes new VP with jitter
  - `Camera::processKeyboard(Window &window, float deltaTime)`
  - `Camera::processMouseMovement(float xPos, float yPos)`
  - `Camera::getCurrentVP() -> glm::mat4`
  - `Camera::getPreviousVP() -> glm::mat4`
  - `Camera::getViewMatrix() -> glm::mat4`
  - `Camera::getProjectionMatrix() -> glm::mat4` (with jitter applied)
  - `Camera::getProjectionMatrixNoJitter() -> glm::mat4`
  - `Camera::getJitterOffset() -> glm::vec2` (in pixel coordinates)
  - `Camera::getPosition() -> glm::vec3`
  - `Camera::setAspectRatio(float)`
  - `Camera::incrementFrameIndex()`

- [ ] **Step 1: Create Camera.h**

Create `src/Camera/Camera.h`:
```cpp
#pragma once
#include <glm/glm.hpp>

class Window;

class Camera {
public:
    void update();
    void processKeyboard(Window &window, float deltaTime);
    void processMouseMovement(float xPos, float yPos);

    glm::mat4 getCurrentVP() const;
    glm::mat4 getPreviousVP() const;
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    glm::mat4 getProjectionMatrixNoJitter() const;
    glm::vec2 getJitterOffset() const;
    glm::vec3 getPosition() const;
    void setAspectRatio(float aspectRatio);
    void incrementFrameIndex();

private:
    // position & orientation
    glm::vec3 m_position  {0.0f, 0.0f, 3.0f};
    glm::vec3 m_front     {0.0f, 0.0f, -1.0f};
    glm::vec3 m_up        {0.0f, 1.0f, 0.0f};
    float     m_yaw       {-90.0f};
    float     m_pitch     {0.0f};

    // mouse state
    float m_lastMouseX{};
    float m_lastMouseY{};
    bool  m_firstMouse{true};

    // projection
    float m_fov        {45.0f};
    float m_aspectRatio{16.0f / 9.0f};
    float m_nearPlane  {0.1f};
    float m_farPlane   {100.0f};

    // temporal state
    glm::mat4 m_currentVP {1.0f};
    glm::mat4 m_previousVP{1.0f};
    uint32_t  m_frameIndex{0};

    // jitter helpers
    float halton(uint32_t index, uint32_t base) const;
    glm::vec2 computeJitter() const;
};
```

- [ ] **Step 2: Create Camera.cpp**

Create `src/Camera/Camera.cpp`:
```cpp
#include "Camera.h"
#include "Window/Window.h"
#include "Config.h"
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <cmath>
#include <algorithm>
#include <GLFW/glfw3.h>

void Camera::update() {
    m_previousVP = m_currentVP;

    glm::mat4 view{glm::lookAt(m_position, m_position + m_front, m_up)};
    glm::mat4 proj{getProjectionMatrix()};

    m_currentVP = proj * view;
}

void Camera::processKeyboard(Window &window, float deltaTime) {
    const float cameraSpeed{2.5f * deltaTime};
    if (window.isKeyPressed(GLFW_KEY_W))
        m_position += cameraSpeed * m_front;
    if (window.isKeyPressed(GLFW_KEY_S))
        m_position -= cameraSpeed * m_front;
    if (window.isKeyPressed(GLFW_KEY_A))
        m_position -= glm::normalize(glm::cross(m_front, m_up)) * cameraSpeed;
    if (window.isKeyPressed(GLFW_KEY_D))
        m_position += glm::normalize(glm::cross(m_front, m_up)) * cameraSpeed;
}

void Camera::processMouseMovement(float xPos, float yPos) {
    if (m_firstMouse) {
        m_lastMouseX = xPos;
        m_lastMouseY = yPos;
        m_firstMouse = false;
    }

    float xOffset{xPos - m_lastMouseX};
    float yOffset{m_lastMouseY - yPos};
    m_lastMouseX = xPos;
    m_lastMouseY = yPos;

    constexpr float sensitivity{0.05f};
    xOffset *= sensitivity;
    yOffset *= sensitivity;

    m_yaw += xOffset;
    m_pitch += yOffset;
    m_pitch = std::clamp(m_pitch, -89.0f, 89.0f);

    glm::vec3 front{
        cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch)),
        sin(glm::radians(m_pitch)),
        sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch))
    };
    m_front = glm::normalize(front);
}

glm::mat4 Camera::getCurrentVP() const {
    return m_currentVP;
}

glm::mat4 Camera::getPreviousVP() const {
    return m_previousVP;
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(m_position, m_position + m_front, m_up);
}

glm::mat4 Camera::getProjectionMatrix() const {
    glm::mat4 proj{getProjectionMatrixNoJitter()};

    // Apply sub-pixel jitter for TAA/upscaling
    glm::vec2 jitter{computeJitter()};
    // Convert pixel jitter to NDC offset
    float internalWidth  {Config::width  * Config::upscaleRatio};
    float internalHeight {Config::height * Config::upscaleRatio};
    proj[2][0] += jitter.x / internalWidth;
    proj[2][1] += jitter.y / internalHeight;

    return proj;
}

glm::mat4 Camera::getProjectionMatrixNoJitter() const {
    glm::mat4 proj{glm::perspective(
        glm::radians(m_fov),
        m_aspectRatio,
        m_nearPlane,
        m_farPlane
    )};
    proj[1][1] *= -1;  // flip Y for Vulkan
    return proj;
}

glm::vec2 Camera::getJitterOffset() const {
    return computeJitter();
}

glm::vec3 Camera::getPosition() const {
    return m_position;
}

void Camera::setAspectRatio(float aspectRatio) {
    m_aspectRatio = aspectRatio;
}

void Camera::incrementFrameIndex() {
    m_frameIndex++;
}

float Camera::halton(uint32_t index, uint32_t base) const {
    float result{0.0f};
    float fraction{1.0f};
    uint32_t i{index};
    while (i > 0) {
        fraction /= static_cast<float>(base);
        result += fraction * static_cast<float>(i % base);
        i /= base;
    }
    return result;
}

glm::vec2 Camera::computeJitter() const {
    // Halton sequence (base 2, 3) centered around 0
    uint32_t idx{(m_frameIndex % Config::jitterSequenceLen) + 1};
    return {
        halton(idx, 2) - 0.5f,
        halton(idx, 3) - 0.5f
    };
}
```

- [ ] **Step 3: Verify — print jitter values**

Temporarily add to main.cpp:
```cpp
#include "Camera/Camera.h"
#include <print>
// inside main loop:
Camera camera{};
for (int i = 0; i < 16; i++) {
    camera.incrementFrameIndex();
    auto jitter = camera.getJitterOffset();
    std::println("Frame {}: jitter ({}, {})", i, jitter.x, jitter.y);
}
```

Expected: 16 different sub-pixel offsets in the [-0.5, 0.5] range, distributed quasi-randomly.

- [ ] **Step 4: Commit**

```bash
git add src/Camera/
git commit -m "feat: add Camera module with jitter, previous-frame VP tracking"
```

---

### Task 4: Renderer — Vulkan Init, Swapchain, Forward Pipeline

**Files:**
- Create: `src/Renderer/Renderer.h`
- Create: `src/Renderer/Renderer.cpp`
- Create: `src/Vertex.h`
- Create: `src/Material.h`
- Create: `src/PushConstants.h`
- Create: `shaders/forward.slang`

**Interfaces:**
- Consumes: `Window`, `Camera`, `Config`
- Produces:
  - `Renderer::initVulkan(Window&)` — full Vulkan setup
  - `Renderer::drawFrame(Window&, Camera&)` — records commands, submits, presents
  - `Renderer::uploadModel(std::vector<Vertex>&, std::vector<uint32_t>&)`
  - `Renderer::uploadTexture(const std::string&, VkFormat) -> uint32_t`
  - `Renderer::uploadMaterials(std::vector<Material>&)`
  - `Renderer::getDevice() -> VkDevice`
  - `Renderer::getPhysicalDevice() -> VkPhysicalDevice`
  - `Renderer::getAllocator() -> VmaAllocator`
  - `Renderer::getQueue() -> VkQueue`
  - `Renderer::getQueueIndex() -> uint32_t`
  - `Renderer::getCommandPool() -> VkCommandPool`
  - `Renderer::getSwapchainExtent() -> VkExtent2D`
  - `Renderer::getSwapchainFormat() -> VkFormat`
  - `Renderer::cleanup()`
  - `Vertex{position, normal, texCoords, materialIndex}`
  - `Material{albedoIndex, normalIndex}`
  - `PushConstants{materialBufferAddress, vertexBufferAddress, model, view, projection, prevViewProjection, jitterOffset}`

This task closely follows the reference project's Renderer. The key differences are:
1. Camera is passed in rather than owned
2. PushConstants includes `prevViewProjection` and `jitterOffset` for later motion vector use
3. Vulkan 1.3 features instead of 1.4 (MoltenVK compatibility)
4. `VK_KHR_portability_subset` extension enabled for MoltenVK
5. `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR` and `VK_KHR_portability_enumeration` for macOS

- [ ] **Step 1: Create Vertex.h**

Create `src/Vertex.h`:
```cpp
#pragma once
#include <glm/glm.hpp>
#include <cstdint>

struct Vertex {
    glm::vec3 position    {};
    glm::vec3 normal      {};
    glm::vec2 texCoords   {};
    uint32_t  materialIndex{};
};
```

- [ ] **Step 2: Create Material.h**

Create `src/Material.h`:
```cpp
#pragma once
#include <cstdint>

struct Material {
    uint32_t albedoIndex{};
    uint32_t normalIndex{};
};
```

- [ ] **Step 3: Create PushConstants.h**

Create `src/PushConstants.h`:
```cpp
#pragma once
#include "volk.h"
#include <glm/glm.hpp>

struct PushConstants {
    VkDeviceAddress materialBufferAddress;
    VkDeviceAddress vertexBufferAddress;
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
    alignas(16) glm::mat4 prevViewProjection;
    alignas(16) glm::vec2 jitterOffset;
};
```

- [ ] **Step 4: Create Renderer.h**

Create `src/Renderer/Renderer.h`:
```cpp
#pragma once
#include "volk.h"
#include "vk_mem_alloc.h"
#include <vector>
#include <cstdint>
#include <string>
class Window;
class Camera;
struct Vertex;
struct Material;

class Renderer {
public:
    void initVulkan(Window &window);
    void drawFrame(Window &window, Camera &camera);
    void uploadModel(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices);
    uint32_t uploadTexture(const std::string &path, VkFormat format);
    void uploadMaterials(std::vector<Material> &materials);
    VkDevice getDevice() const;
    VkPhysicalDevice getPhysicalDevice() const;
    VmaAllocator getAllocator() const;
    VkQueue getQueue() const;
    uint32_t getQueueIndex() const;
    VkCommandPool getCommandPool() const;
    VkExtent2D getSwapchainExtent() const;
    VkFormat getSwapchainFormat() const;
    void cleanup();

private:
    VkInstance                   m_instance                {VK_NULL_HANDLE};
    VkPhysicalDevice             m_physicalDevice          {VK_NULL_HANDLE};
    VkDevice                     m_device                  {VK_NULL_HANDLE};
    VkQueue                      m_queue                   {VK_NULL_HANDLE};
    uint32_t                     m_queueIndex              {};
    VmaAllocator                 m_allocator               {VK_NULL_HANDLE};
    VkSurfaceKHR                 m_surface                 {VK_NULL_HANDLE};
    // swapchain
    VkSwapchainKHR               m_swapchain               {VK_NULL_HANDLE};
    VkExtent2D                   m_swapchainExtent         {};
    VkSurfaceFormatKHR           m_swapchainSurfaceFormat  {};
    std::vector<VkImage>         m_swapchainImages         {};
    std::vector<VkImageView>     m_swapchainImageViews     {};
    // depth testing
    VkImage                      m_depthImage              {VK_NULL_HANDLE};
    VkImageView                  m_depthImageView          {VK_NULL_HANDLE};
    VmaAllocation                m_depthImageAllocation    {VK_NULL_HANDLE};
    // pipeline & rendering
    VkPipelineLayout             m_pipelineLayout          {VK_NULL_HANDLE};
    VkPipeline                   m_graphicsPipeline        {VK_NULL_HANDLE};
    VkCommandPool                m_commandPool             {VK_NULL_HANDLE};
    std::vector<VkCommandBuffer> m_commandBuffers          {};
    std::vector<VkFence>         m_renderWaitFences        {};
    std::vector<VkSemaphore>     m_acquireImageSemaphores  {};
    std::vector<VkSemaphore>     m_renderFinishedSemaphores{};
    uint32_t                     m_frameIndex              {};
    // model data
    VkBuffer                     m_vertexBuffer            {VK_NULL_HANDLE};
    VmaAllocation                m_vertexAllocation        {VK_NULL_HANDLE};
    VmaAllocationInfo            m_vertexAllocationInfo    {};
    VkDeviceAddress              m_vertexBufferAddress     {};
    VkBuffer                     m_indexBuffer             {VK_NULL_HANDLE};
    uint32_t                     m_indexCount              {};
    VmaAllocation                m_indexAllocation         {VK_NULL_HANDLE};
    VmaAllocationInfo            m_indexAllocationInfo     {};
    // descriptors
    VkDescriptorSetLayout        m_descriptorSetLayout     {VK_NULL_HANDLE};
    VkDescriptorPool             m_descriptorPool          {VK_NULL_HANDLE};
    VkDescriptorSet              m_descriptorSet           {VK_NULL_HANDLE};
    // textures
    VkSampler                    m_textureSampler          {VK_NULL_HANDLE};
    std::vector<VkImage>         m_textureImages           {};
    std::vector<VmaAllocation>   m_textureAllocations      {};
    std::vector<VkImageView>     m_textureImageViews       {};
    // materials
    VkBuffer                     m_materialBuffer          {VK_NULL_HANDLE};
    VmaAllocation                m_materialAllocation      {VK_NULL_HANDLE};
    VmaAllocationInfo            m_materialAllocationInfo  {};
    VkDeviceAddress              m_materialBufferAddress   {};
    // extensions
    std::vector<const char *>    m_requiredDeviceExtensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        "VK_KHR_portability_subset",
    };

    void createInstance(Window &window);
    void pickPhysicalDevice();
    void createDevice();
    void createAllocator();
    void createSurface(Window &window);
    void createSwapchain(Window &window);
    void createImageViews();
    void createDepthResources();
    void createGraphicsPipeline();
    void createCommandPool();
    void createCommandBuffers();
    void createSyncObjects();
    void createBuffer(VkBuffer &buffer, VkDeviceSize size, VkBufferUsageFlags usage, VmaAllocation &allocation, VmaAllocationInfo &allocationInfo);
    void createDescriptorSet();
    void createTextureSampler();
    void recreateSwapchain(Window &window);

    std::vector<char> readFile(const std::string &filename);
    void checkResult(VkResult result, const char *errorMessage) const;
    void swapchainCleanup();
    void syncObjectCleanup();
    void vmaCleanup();
};
```

- [ ] **Step 5: Create Renderer.cpp**

Create `src/Renderer/Renderer.cpp`. This follows the reference project's implementation closely. Key adaptations for MoltenVK:

1. `createInstance`: Add `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR` flag and `VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME` + `VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME` to instance extensions. Use `VK_API_VERSION_1_3` instead of 1.4.

2. `createDevice`: Enable Vulkan 1.2 features (descriptorIndexing, bufferDeviceAddress, scalarBlockLayout, runtimeDescriptorArray, shaderSampledImageArrayNonUniformIndexing, descriptorBindingPartiallyBound) and Vulkan 1.3 features (synchronization2, dynamicRendering). No 1.4 features (no hostImageCopy, no unifiedImageLayouts).

3. `uploadTexture`: Use staging buffer + command buffer copy instead of host image copy (since hostImageCopy is not available on MoltenVK). This means:
   - Create a staging buffer with `VMA_MEMORY_USAGE_AUTO` + `VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT`
   - Copy pixel data to staging buffer
   - Record a command buffer that transitions image layout with `vkCmdPipelineBarrier2`, then `vkCmdCopyBufferToImage`
   - Submit and wait

4. `drawFrame`: Pass `Camera` for view/projection matrices instead of computing them internally. Push `prevViewProjection` and `jitterOffset` in PushConstants.

```cpp
#include "Renderer.h"
#include "PushConstants.h"
#include "Config.h"
#include "Window/Window.h"
#include "Camera/Camera.h"
#include "volk.h"
#include "vk_mem_alloc.h"
#include "stb_image.h"
#include "Vertex.h"
#include "Material.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <ranges>
#include <string>
#include <fstream>
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <vector>
#include <queue>
#include <print>
#include <cstring>
#include <cmath>
#include <array>

void Renderer::initVulkan(Window &window) {
    checkResult(volkInitialize(), "Error: failed to initialize Volk");
    createInstance(window);
    pickPhysicalDevice();
    createDevice();
    createAllocator();
    createSurface(window);
    createSwapchain(window);
    createImageViews();
    createDepthResources();
    createDescriptorSet();
    createTextureSampler();
    createGraphicsPipeline();
    createCommandPool();
    createCommandBuffers();
    createSyncObjects();
}

void Renderer::uploadModel(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices) {
    createBuffer(
        m_vertexBuffer,
        vertices.size() * sizeof(Vertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_vertexAllocation,
        m_vertexAllocationInfo
    );
    createBuffer(
        m_indexBuffer,
        indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_indexAllocation,
        m_indexAllocationInfo
    );

    memcpy(m_vertexAllocationInfo.pMappedData, vertices.data(), vertices.size() * sizeof(Vertex));
    memcpy(m_indexAllocationInfo.pMappedData, indices.data(), indices.size() * sizeof(uint32_t));

    VkBufferDeviceAddressInfo bufferDeviceAddressInfo{
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext  = nullptr,
        .buffer = m_vertexBuffer
    };
    m_vertexBufferAddress = vkGetBufferDeviceAddress(m_device, &bufferDeviceAddressInfo);
    m_indexCount = static_cast<uint32_t>(indices.size());
}

uint32_t Renderer::uploadTexture(const std::string &path, VkFormat format) {
    int width{}, height{}, channels{};
    stbi_uc *pixels{stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha)};
    if (pixels == nullptr) {
        throw std::runtime_error(std::format("Error: failed to load texture: {}", path));
    }

    VkDeviceSize imageSize{static_cast<VkDeviceSize>(width * height * 4)};

    // Create staging buffer
    VkBuffer stagingBuffer{VK_NULL_HANDLE};
    VmaAllocation stagingAllocation{VK_NULL_HANDLE};
    VmaAllocationInfo stagingAllocInfo{};
    createBuffer(stagingBuffer, imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        stagingAllocation, stagingAllocInfo);
    memcpy(stagingAllocInfo.pMappedData, pixels, imageSize);
    stbi_image_free(pixels);

    // Create image
    VkImageCreateInfo imageCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = {},
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = format,
        .extent        = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1},
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    VmaAllocationCreateInfo allocationCreateInfo{
        .usage = VMA_MEMORY_USAGE_AUTO
    };

    VkImage image{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    checkResult(vmaCreateImage(m_allocator, &imageCreateInfo, &allocationCreateInfo, &image, &allocation, nullptr), "Error: failed to create texture image");

    // Transition + copy via command buffer
    VkCommandBuffer cmd{};
    VkCommandBufferAllocateInfo cmdAllocInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = m_commandPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    vkAllocateCommandBuffers(m_device, &cmdAllocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Transition to TRANSFER_DST
    VkImageMemoryBarrier2 toTransferBarrier{
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext               = nullptr,
        .srcStageMask        = VK_PIPELINE_STAGE_2_NONE,
        .srcAccessMask       = VK_ACCESS_2_NONE,
        .dstStageMask        = VK_PIPELINE_STAGE_2_COPY_BIT,
        .dstAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = image,
        .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    VkDependencyInfo depInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &toTransferBarrier
    };
    vkCmdPipelineBarrier2(cmd, &depInfo);

    // Copy buffer to image
    VkBufferImageCopy region{
        .bufferOffset      = 0,
        .bufferRowLength   = 0,
        .bufferImageHeight = 0,
        .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset       = {0, 0, 0},
        .imageExtent       = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1}
    };
    vkCmdCopyBufferToImage(cmd, stagingBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition to SHADER_READ_ONLY
    VkImageMemoryBarrier2 toReadBarrier{
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext               = nullptr,
        .srcStageMask        = VK_PIPELINE_STAGE_2_COPY_BIT,
        .srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask        = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dstAccessMask       = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        .oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = image,
        .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    VkDependencyInfo depInfo2{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &toReadBarrier
    };
    vkCmdPipelineBarrier2(cmd, &depInfo2);

    vkEndCommandBuffer(cmd);

    // Submit and wait
    VkCommandBufferSubmitInfo cmdSubmitInfo{
        .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext         = nullptr,
        .commandBuffer = cmd,
        .deviceMask    = 0
    };
    VkSubmitInfo2 submitInfo{
        .sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext                  = nullptr,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos    = &cmdSubmitInfo
    };
    vkQueueSubmit2(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_queue);

    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);
    vmaDestroyBuffer(m_allocator, stagingBuffer, stagingAllocation);

    // Create image view
    VkImageViewCreateInfo imageViewCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = {},
        .image            = image,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = format,
        .components       = {},
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    VkImageView imageView{VK_NULL_HANDLE};
    checkResult(vkCreateImageView(m_device, &imageViewCreateInfo, nullptr, &imageView), "Error: failed to create texture image view");

    // Write into bindless array
    static uint32_t currFreeSlot{};
    uint32_t slot{currFreeSlot++};
    VkDescriptorImageInfo descriptorImageInfo{
        .sampler     = m_textureSampler,
        .imageView   = imageView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    };
    VkWriteDescriptorSet writeDescriptorSet{
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext           = nullptr,
        .dstSet          = m_descriptorSet,
        .dstBinding      = 0,
        .dstArrayElement = slot,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo      = &descriptorImageInfo
    };
    vkUpdateDescriptorSets(m_device, 1, &writeDescriptorSet, 0, nullptr);

    m_textureImages.push_back(image);
    m_textureAllocations.push_back(allocation);
    m_textureImageViews.push_back(imageView);
    return slot;
}

void Renderer::uploadMaterials(std::vector<Material> &materials) {
    createBuffer(
        m_materialBuffer,
        materials.size() * sizeof(Material),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_materialAllocation,
        m_materialAllocationInfo
    );
    memcpy(m_materialAllocationInfo.pMappedData, materials.data(), materials.size() * sizeof(Material));

    VkBufferDeviceAddressInfo bufferDeviceAddressInfo{
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext  = nullptr,
        .buffer = m_materialBuffer
    };
    m_materialBufferAddress = vkGetBufferDeviceAddress(m_device, &bufferDeviceAddressInfo);
}

void Renderer::createInstance(Window &window) {
    constexpr VkApplicationInfo applicationInfo{
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext              = nullptr,
        .pApplicationName   = Config::title,
        .applicationVersion = Config::version,
        .pEngineName        = nullptr,
        .engineVersion      = {},
        .apiVersion         = VK_API_VERSION_1_3,
    };

    auto [glfwExtensionCount, glfwExtensions] = window.getRequiredInstanceExtensions();

    // Merge GLFW extensions with portability enumeration
    std::vector<const char *> instanceExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    const VkInstanceCreateInfo instanceCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext                   = nullptr,
        .flags                   = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
        .pApplicationInfo        = &applicationInfo,
        .enabledLayerCount       = 0,
        .ppEnabledLayerNames     = nullptr,
        .enabledExtensionCount   = static_cast<uint32_t>(instanceExtensions.size()),
        .ppEnabledExtensionNames = instanceExtensions.data()
    };

    checkResult(vkCreateInstance(&instanceCreateInfo, nullptr, &m_instance), "Error: failed to create Vulkan instance");
    volkLoadInstance(m_instance);
}

void Renderer::pickPhysicalDevice() {
    uint32_t physicalDeviceCount{};
    checkResult(vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, nullptr), "Error: failed to enumerate physical device count");
    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    checkResult(vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, physicalDevices.data()), "Error: failed to fill physicalDevices vector");

    std::priority_queue<std::pair<int, VkPhysicalDevice>> priorityQueue{};
    for (VkPhysicalDevice physicalDevice : physicalDevices) {
        VkPhysicalDeviceProperties deviceProperties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

        int score{};
        switch (deviceProperties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                score += 2;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                score += 1;
                break;
            default:
                continue;
        }
        priorityQueue.push({score, physicalDevice});
    }

    if (!priorityQueue.empty()) {
        m_physicalDevice = priorityQueue.top().second;
    } else {
        throw std::runtime_error("Error: failed to find compatible GPU");
    }

    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(m_physicalDevice, &deviceProperties);
    std::println("Selected GPU: {}", deviceProperties.deviceName);
}

void Renderer::createDevice() {
    uint32_t queueFamilyPropertyCount{};
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyPropertyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyPropertyCount, queueFamilyProperties.data());

    for (auto [i, queueFamilyProperty] : std::views::enumerate(queueFamilyProperties)) {
        if (queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            m_queueIndex = i;
            break;
        }
    }

    float queuePriority{1.0f};
    VkDeviceQueueCreateInfo queueCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = {},
        .queueFamilyIndex = m_queueIndex,
        .queueCount       = 1,
        .pQueuePriorities = &queuePriority
    };

    VkPhysicalDeviceVulkan11Features vulkan11Features{
        .sType                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext                = nullptr,
        .shaderDrawParameters = VK_TRUE
    };

    VkPhysicalDeviceVulkan12Features vulkan12Features{
        .sType                                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext                                     = &vulkan11Features,
        .descriptorIndexing                        = VK_TRUE,
        .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
        .descriptorBindingPartiallyBound           = VK_TRUE,
        .runtimeDescriptorArray                    = VK_TRUE,
        .scalarBlockLayout                         = VK_TRUE,
        .bufferDeviceAddress                       = VK_TRUE,
    };

    VkPhysicalDeviceVulkan13Features vulkan13Features{
        .sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext            = &vulkan12Features,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE
    };

    VkDeviceCreateInfo deviceCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &vulkan13Features,
        .queueCreateInfoCount    = 1,
        .pQueueCreateInfos       = &queueCreateInfo,
        .enabledExtensionCount   = static_cast<uint32_t>(m_requiredDeviceExtensions.size()),
        .ppEnabledExtensionNames = m_requiredDeviceExtensions.data(),
        .pEnabledFeatures        = nullptr
    };

    checkResult(vkCreateDevice(m_physicalDevice, &deviceCreateInfo, nullptr, &m_device), "Error: failed to create Vulkan logical device");
    vkGetDeviceQueue(m_device, m_queueIndex, 0, &m_queue);
    volkLoadDevice(m_device);
}

void Renderer::createAllocator() {
    VmaAllocatorCreateInfo allocatorCreateInfo{
        .flags                       = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice              = m_physicalDevice,
        .device                      = m_device,
        .preferredLargeHeapBlockSize = {},
        .pAllocationCallbacks        = nullptr,
        .pDeviceMemoryCallbacks      = nullptr,
        .pHeapSizeLimit              = nullptr,
        .pVulkanFunctions            = {},
        .instance                    = m_instance,
        .vulkanApiVersion            = VK_API_VERSION_1_3,
    };

    VmaVulkanFunctions vulkanFunctions{};
    checkResult(vmaImportVulkanFunctionsFromVolk(&allocatorCreateInfo, &vulkanFunctions), "Error: failed to import Vulkan functions for VMA from Volk");
    allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;

    checkResult(vmaCreateAllocator(&allocatorCreateInfo, &m_allocator), "Error: Failed to create VMA Allocator");
}

// createSurface, createSwapchain, createImageViews, createDepthResources,
// createGraphicsPipeline, createCommandPool, createCommandBuffers, createSyncObjects,
// createBuffer, createDescriptorSet, createTextureSampler — follow the reference
// project's implementations with the adaptations noted above.
// drawFrame passes Camera matrices to PushConstants.
// cleanup, swapchainCleanup, syncObjectCleanup, vmaCleanup, readFile, checkResult,
// recreateSwapchain — same as reference project.
// Accessor methods are trivial one-liners.

// (The full implementation of each function is provided in the reference project's
// Renderer.cpp. Copy each function, applying the MoltenVK adaptations:
// - Use VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL instead of VK_IMAGE_LAYOUT_GENERAL for textures
// - Use staging buffer for texture uploads
// - Use VK_API_VERSION_1_3
// - Pass Camera to drawFrame for view/projection/prevVP/jitter)
```

The `drawFrame` method's key difference from the reference — the push constants section:

```cpp
void Renderer::drawFrame(Window &window, Camera &camera) {
    // ... (fence wait, image acquire, command buffer begin, barriers — same as reference) ...

    // Inside the rendering section, replace the reference's matrix computation with:
    glm::mat4 model{glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(0.01f)), glm::vec3(0.0f, -100.0f, 0.0f))};
    PushConstants pushConstants{
        .materialBufferAddress = m_materialBufferAddress,
        .vertexBufferAddress   = m_vertexBufferAddress,
        .model                 = model,
        .view                  = camera.getViewMatrix(),
        .projection            = camera.getProjectionMatrix(),
        .prevViewProjection    = camera.getPreviousVP(),
        .jitterOffset          = camera.getJitterOffset()
    };

    // ... (push constants, draw, end rendering, present — same as reference) ...
}
```

- [ ] **Step 6: Create forward.slang**

Create `shaders/forward.slang`:
```hlsl
[[vk::binding(0, 0)]]
Sampler2D g_textures[];

struct Vertex {
    float3 position;
    float3 normal;
    float2 texCoord;
    uint32_t materialIndex;
};

struct Material {
    uint albedoIndex;
    uint normalIndex;
};

struct PushConstants {
    Material *materialBuffer;
    Vertex *vertexBuffer;
    float4x4 model;
    float4x4 view;
    float4x4 projection;
    float4x4 prevViewProjection;
    float2 jitterOffset;
};

struct VertexOutput {
    float4 position : SV_Position;
    float3 normal;
    float2 texCoord;
    float3 lightDir;
    float3 viewDir;
    nointerpolation uint materialIndex;
};

[[vk::push_constant]]
PushConstants pushConstants;

const static uint g_invalidTextureIndex = 0xFFFFFFFF;
const static float g_ambientStrength = 0.1;
const static float3 g_lightColor = {1.0, 1.0, 1.0};
const static float4 g_lightPos = {-2.5, 4.0, 5.0, 1.0};
const static float g_shininess = 32.0;
const static float g_specularStrength = 0.5;

[shader("vertex")]
VertexOutput vertexMain(uint vertexIndex: SV_VertexID) {
    Vertex vertex = pushConstants.vertexBuffer[vertexIndex];
    float4x4 model = pushConstants.model;
    float4x4 view = pushConstants.view;
    float4x4 projection = pushConstants.projection;

    float4 worldPos = mul(model, float4(vertex.position, 1.0));
    float4 viewPos = mul(view, worldPos);

    VertexOutput output;
    output.position = mul(projection, viewPos);
    output.normal   = mul((float3x3)model, vertex.normal);
    output.texCoord = vertex.texCoord;
    output.lightDir = g_lightPos.xyz - worldPos.xyz;
    output.viewDir  = -viewPos.xyz;
    output.materialIndex = vertex.materialIndex;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput vertex) : SV_Target {
    Material material = pushConstants.materialBuffer[vertex.materialIndex];

    float4 baseColor = float4(1.0, 1.0, 1.0, 1.0);
    if (material.albedoIndex != g_invalidTextureIndex) {
        baseColor = g_textures[NonUniformResourceIndex(material.albedoIndex)].Sample(vertex.texCoord);
    }

    if (baseColor.a < 0.1) {
        discard;
    }

    float3 albedo = baseColor.rgb;

    float3 normal = normalize(vertex.normal);
    float3 lightDir = normalize(vertex.lightDir);
    float3 viewDir = normalize(vertex.viewDir);

    float3 ambient = g_ambientStrength * g_lightColor * albedo;

    float diffuseFactor = max(dot(normal, lightDir), 0.0);
    float3 diffuse = diffuseFactor * g_lightColor * albedo;

    float3 halfwayDir = normalize(lightDir + viewDir);
    float specularFactor = pow(max(dot(normal, halfwayDir), 0.0), g_shininess);
    float3 specular = g_specularStrength * specularFactor * g_lightColor;

    float3 result = ambient + specular + diffuse;
    return float4(result, 1.0);
}
```

- [ ] **Step 7: Verify — render a clear color**

Before loading a model, verify the pipeline initializes without errors and clears to black. The window should show a black screen with no validation errors.

- [ ] **Step 8: Commit**

```bash
git add src/Renderer/ src/Vertex.h src/Material.h src/PushConstants.h shaders/forward.slang
git commit -m "feat: add Vulkan renderer with forward pipeline (MoltenVK compatible)"
```

---

### Task 5: Model Loader & App Integration

**Files:**
- Create: `src/ModelLoader/ModelLoader.h`
- Create: `src/ModelLoader/ModelLoader.cpp`
- Create: `src/App/App.h`
- Create: `src/App/App.cpp`
- Modify: `src/main.cpp`

**Interfaces:**
- Consumes: `Renderer::uploadModel()`, `Renderer::uploadTexture()`, `Renderer::uploadMaterials()`, `Window`, `Input`, `Camera`, `Vertex`, `Material`
- Produces:
  - `ModelLoader::loadModel(std::vector<Vertex>&, std::vector<uint32_t>&, std::vector<Material>&, Renderer&)`
  - `App::run()` — full application lifecycle

- [ ] **Step 1: Create ModelLoader.h**

Create `src/ModelLoader/ModelLoader.h`:
```cpp
#pragma once
#include <vector>
#include <cstdint>
struct Vertex;
struct Material;
class Renderer;

class ModelLoader {
public:
    void loadModel(
        std::vector<Vertex> &vertices,
        std::vector<uint32_t> &indices,
        std::vector<Material> &materials,
        Renderer &renderer
    );
};
```

- [ ] **Step 2: Create ModelLoader.cpp**

Create `src/ModelLoader/ModelLoader.cpp` — same as reference project's implementation. Loads glTF via assimp, extracts vertices/indices/materials, uploads textures via `Renderer::uploadTexture()`.

```cpp
#include "ModelLoader.h"
#include "Renderer/Renderer.h"
#include "Vertex.h"
#include "Material.h"
#include "volk.h"
#include <filesystem>
#include <stdexcept>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/mesh.h>
#include <ranges>
#include <cstdint>
#include <format>
#include <string>
#include <vector>
#include <unordered_map>

void ModelLoader::loadModel(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices, std::vector<Material> &materials, Renderer &renderer) {
    Assimp::Importer importer{};
    const char *path{PROJECT_ROOT_DIR "models/scene.gltf"};
    const aiScene *scene{importer.ReadFile(
        path,
        aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_FlipUVs
    )};

    if (scene == nullptr || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::string error{std::format("Assimp error: {}\n", importer.GetErrorString())};
        throw std::runtime_error(error);
    }

    std::filesystem::path modelDir{std::filesystem::path(path).parent_path()};
    std::unordered_map<std::string, uint32_t> texturePathToSlot{};
    std::unordered_map<uint32_t, uint32_t> assimpMatToCompact{};

    for (uint32_t i : std::views::iota(0u, scene->mNumMeshes)) {
        const aiMesh *mesh{scene->mMeshes[i]};

        uint32_t assimpMatIndex{mesh->mMaterialIndex};
        uint32_t compactMatIndex{};

        auto matIt{assimpMatToCompact.find(assimpMatIndex)};
        if (matIt != assimpMatToCompact.end()) {
            compactMatIndex = matIt->second;
        } else {
            const aiMaterial *material{scene->mMaterials[assimpMatIndex]};

            uint32_t albedoIndex{0xFFFFFFFF};
            aiString albedoPath{};
            if (material->GetTexture(aiTextureType_BASE_COLOR, 0, &albedoPath) != AI_SUCCESS) {
                material->GetTexture(aiTextureType_DIFFUSE, 0, &albedoPath);
            }
            if (albedoPath.length > 0) {
                std::string fullPath{(modelDir / albedoPath.C_Str()).string()};
                auto it{texturePathToSlot.find(fullPath)};
                if (it != texturePathToSlot.end()) {
                    albedoIndex = it->second;
                } else {
                    albedoIndex = renderer.uploadTexture(fullPath, VK_FORMAT_R8G8B8A8_SRGB);
                    texturePathToSlot.emplace(fullPath, albedoIndex);
                }
            }

            uint32_t normalIndex{0xFFFFFFFF};
            aiString normalPath{};
            if (material->GetTexture(aiTextureType_NORMALS, 0, &normalPath) == AI_SUCCESS) {
                std::string fullPath{(modelDir / normalPath.C_Str()).string()};
                auto it{texturePathToSlot.find(fullPath)};
                if (it != texturePathToSlot.end()) {
                    normalIndex = it->second;
                } else {
                    normalIndex = renderer.uploadTexture(fullPath, VK_FORMAT_R8G8B8A8_UNORM);
                    texturePathToSlot.emplace(fullPath, normalIndex);
                }
            }

            compactMatIndex = static_cast<uint32_t>(materials.size());
            materials.push_back(Material{.albedoIndex = albedoIndex, .normalIndex = normalIndex});
            assimpMatToCompact.emplace(assimpMatIndex, compactMatIndex);
        }

        uint32_t vertexOffset{static_cast<uint32_t>(vertices.size())};
        for (uint32_t j : std::views::iota(0u, mesh->mNumVertices)) {
            Vertex vertex{};
            vertex.position = {mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z};

            if (mesh->HasNormals()) {
                vertex.normal = {mesh->mNormals[j].x, mesh->mNormals[j].y, mesh->mNormals[j].z};
            }
            if (mesh->HasTextureCoords(0)) {
                vertex.texCoords = {mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y};
            }

            vertex.materialIndex = compactMatIndex;
            vertices.push_back(vertex);
        }

        for (uint32_t j : std::views::iota(0u, mesh->mNumFaces)) {
            const aiFace &face{mesh->mFaces[j]};
            for (uint32_t k : std::views::iota(0u, face.mNumIndices)) {
                indices.push_back(vertexOffset + face.mIndices[k]);
            }
        }
    }
}
```

- [ ] **Step 3: Create App.h**

Create `src/App/App.h`:
```cpp
#pragma once
#include "Renderer/Renderer.h"
#include "Window/Window.h"
#include "Input/Input.h"
#include "Camera/Camera.h"
#include "ModelLoader/ModelLoader.h"
#include "Material.h"
#include "Vertex.h"
#include <cstdint>
#include <vector>

class App {
public:
    void run();

private:
    Window                m_window{};
    Input                 m_input{};
    Renderer              m_renderer{};
    Camera                m_camera{};
    ModelLoader           m_modelLoader{};
    std::vector<Vertex>   m_vertices{};
    std::vector<uint32_t> m_indices{};
    std::vector<Material> m_materials{};
    float                 m_deltaTime{0.0f};
    float                 m_lastFrame{0.0f};

    void mainLoop();
    void cleanup();
};
```

- [ ] **Step 4: Create App.cpp**

Create `src/App/App.cpp`:
```cpp
#include "App.h"

void App::run() {
    m_window.initWindow();
    m_window.captureCursor();

    // Set up mouse callback — store Camera pointer in GLFW user pointer
    glfwSetWindowUserPointer(m_window.getWindow(), &m_camera);
    glfwSetCursorPosCallback(m_window.getWindow(), [](GLFWwindow *window, double xPos, double yPos) {
        auto *camera{static_cast<Camera *>(glfwGetWindowUserPointer(window))};
        camera->processMouseMovement(static_cast<float>(xPos), static_cast<float>(yPos));
    });

    m_renderer.initVulkan(m_window);
    m_camera.setAspectRatio(
        static_cast<float>(m_renderer.getSwapchainExtent().width) /
        static_cast<float>(m_renderer.getSwapchainExtent().height)
    );

    m_modelLoader.loadModel(m_vertices, m_indices, m_materials, m_renderer);
    m_renderer.uploadModel(m_vertices, m_indices);
    m_renderer.uploadMaterials(m_materials);

    mainLoop();
    cleanup();
}

void App::mainLoop() {
    while (!m_window.shouldClose()) {
        float currentFrame{m_window.getTime()};
        m_deltaTime = currentFrame - m_lastFrame;
        m_lastFrame = currentFrame;

        m_window.pollEvents();
        m_input.processUserInput(m_window);
        m_camera.processKeyboard(m_window, m_deltaTime);
        m_camera.update();

        m_renderer.drawFrame(m_window, m_camera);

        m_camera.incrementFrameIndex();
    }
}

void App::cleanup() {
    m_renderer.cleanup();
    m_window.cleanup();
}
```

- [ ] **Step 5: Update main.cpp**

Replace `src/main.cpp`:
```cpp
#include "App/App.h"
#include <iostream>
#include <exception>

int main() {
    try {
        App app{};
        app.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}
```

- [ ] **Step 6: Download a test model**

Download a glTF model to `models/`. Either use the same Cloud Strife model as the reference project or pick another glTF model. Update the path in `ModelLoader.cpp` accordingly.

- [ ] **Step 7: Verify — rendered model on screen**

Build and run. Expected: a lit 3D model rendered on screen. Camera moves with WASD, mouse look works. The scene should have slight sub-pixel jitter (barely visible) due to the Halton sequence applied to the projection matrix.

- [ ] **Step 8: Commit**

```bash
git add src/App/ src/ModelLoader/ src/main.cpp
git commit -m "feat: add ModelLoader, App integration — full forward renderer working"
```

---

### Task 6: Motion Vectors (Velocity Buffer)

**Files:**
- Modify: `shaders/forward.slang` — add velocity output to fragment shader
- Create: `src/MotionVectors/MotionVectors.h`
- Create: `src/MotionVectors/MotionVectors.cpp`
- Modify: `src/Renderer/Renderer.h` — add velocity render target
- Modify: `src/Renderer/Renderer.cpp` — create velocity image, add as second color attachment
- Modify: `CMakeLists.txt` — add MotionVectors source files

**Interfaces:**
- Consumes: `Renderer` (device, allocator, swapchain extent), `Camera` (currentVP, previousVP)
- Produces:
  - `MotionVectors::init(Renderer&)` — creates velocity image (RG16F), image view
  - `MotionVectors::getImageView() -> VkImageView`
  - `MotionVectors::getImage() -> VkImage`
  - `MotionVectors::cleanup(Renderer&)`
  - Velocity data written in forward.slang fragment shader as second color output

- [ ] **Step 1: Create MotionVectors.h**

Create `src/MotionVectors/MotionVectors.h`:
```cpp
#pragma once
#include "volk.h"
#include "vk_mem_alloc.h"

class Renderer;

class MotionVectors {
public:
    void init(Renderer &renderer);
    VkImageView getImageView() const;
    VkImage getImage() const;
    void cleanup(Renderer &renderer);

private:
    VkImage       m_velocityImage     {VK_NULL_HANDLE};
    VkImageView   m_velocityImageView {VK_NULL_HANDLE};
    VmaAllocation m_velocityAllocation{VK_NULL_HANDLE};
};
```

- [ ] **Step 2: Create MotionVectors.cpp**

Create `src/MotionVectors/MotionVectors.cpp`:
```cpp
#include "MotionVectors.h"
#include "Renderer/Renderer.h"
#include "Config.h"

void MotionVectors::init(Renderer &renderer) {
    VkExtent2D extent{renderer.getSwapchainExtent()};

    VkImageCreateInfo imageCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = {},
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = VK_FORMAT_R16G16_SFLOAT,
        .extent        = {extent.width, extent.height, 1},
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    VmaAllocationCreateInfo allocationCreateInfo{
        .usage = VMA_MEMORY_USAGE_AUTO
    };

    vmaCreateImage(renderer.getAllocator(), &imageCreateInfo, &allocationCreateInfo,
        &m_velocityImage, &m_velocityAllocation, nullptr);

    VkImageViewCreateInfo imageViewCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = {},
        .image            = m_velocityImage,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = VK_FORMAT_R16G16_SFLOAT,
        .components       = {},
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };

    vkCreateImageView(renderer.getDevice(), &imageViewCreateInfo, nullptr, &m_velocityImageView);
}

VkImageView MotionVectors::getImageView() const {
    return m_velocityImageView;
}

VkImage MotionVectors::getImage() const {
    return m_velocityImage;
}

void MotionVectors::cleanup(Renderer &renderer) {
    vkDestroyImageView(renderer.getDevice(), m_velocityImageView, nullptr);
    vmaDestroyImage(renderer.getAllocator(), m_velocityImage, m_velocityAllocation);
}
```

- [ ] **Step 3: Modify forward.slang — add velocity output**

Add a second render target output for velocity. Update the vertex shader to compute previous-frame clip position and pass it to the fragment shader. Update the fragment shader to output motion vectors:

```hlsl
// Add to VertexOutput:
    float4 currentClipPos : CLIP_POS;
    float4 previousClipPos : PREV_CLIP_POS;

// In vertexMain, after computing output.position:
    output.currentClipPos = output.position;
    float4 prevClipPos = mul(pushConstants.prevViewProjection, mul(model, float4(vertex.position, 1.0)));
    output.previousClipPos = prevClipPos;

// Add second fragment output struct:
struct FragmentOutput {
    float4 color    : SV_Target0;
    float2 velocity : SV_Target1;
};

// Change fragmentMain return type and add velocity computation:
[shader("fragment")]
FragmentOutput fragmentMain(VertexOutput vertex) {
    // ... existing lighting code ...

    FragmentOutput output;
    output.color = float4(result, 1.0);

    // Compute screen-space motion vector
    float2 currentNDC = vertex.currentClipPos.xy / vertex.currentClipPos.w;
    float2 previousNDC = vertex.previousClipPos.xy / vertex.previousClipPos.w;
    output.velocity = (currentNDC - previousNDC) * 0.5; // NDC [-1,1] to [-0.5,0.5]

    return output;
}
```

- [ ] **Step 4: Modify Renderer — add velocity as second color attachment**

In `Renderer::drawFrame()`, add the velocity image view as a second color attachment in the `VkRenderingInfo`. In `createGraphicsPipeline()`, add a second `VkPipelineColorBlendAttachmentState` and set `colorAttachmentCount = 2` with formats `{swapchainFormat, VK_FORMAT_R16G16_SFLOAT}` in `VkPipelineRenderingCreateInfo`.

Add a barrier for the velocity image alongside the color/depth barriers. Add the velocity image to the `VkRenderingInfo` color attachments array.

The Renderer needs to accept the `MotionVectors` object (or its image view) — add a method:
```cpp
void Renderer::setVelocityTarget(VkImageView velocityImageView, VkImage velocityImage);
```

Or pass it via `drawFrame`. Store it as a member.

- [ ] **Step 5: Update CMakeLists.txt**

Add to the SOURCE list:
```cmake
src/MotionVectors/MotionVectors.cpp
```

- [ ] **Step 6: Integrate in App**

Add `MotionVectors m_motionVectors{};` to App. Initialize after renderer init, pass to renderer.

- [ ] **Step 7: Verify — motion vectors written**

Use a validation layer or RenderDoc (if available) to inspect the velocity render target. When the camera is stationary, velocity should be near zero. When the camera moves, velocity should show non-zero values proportional to screen-space motion.

Alternatively: temporarily modify the fragment shader to visualize velocity as color (`output.color = float4(abs(velocity) * 10.0, 0.0, 1.0)`) and confirm you see colored streaks when the camera moves.

- [ ] **Step 8: Commit**

```bash
git add src/MotionVectors/ shaders/forward.slang src/Renderer/ CMakeLists.txt src/App/
git commit -m "feat: add velocity buffer output for motion vectors"
```

---

### Task 7: TAA Pass

**Files:**
- Create: `src/PostProcess/PostProcessPipeline.h`
- Create: `src/PostProcess/PostProcessPipeline.cpp`
- Create: `src/PostProcess/TAAPass.h`
- Create: `src/PostProcess/TAAPass.cpp`
- Create: `shaders/taa.slang`
- Modify: `src/Renderer/Renderer.h` — render to offscreen color target instead of swapchain
- Modify: `src/Renderer/Renderer.cpp` — offscreen color target, expose for post-process read
- Modify: `src/App/App.cpp` — integrate PostProcessPipeline
- Modify: `CMakeLists.txt` — add sources and shader compilation

**Interfaces:**
- Consumes: `Renderer` (device, allocator, queue, command pool, swapchain extent), color image, depth image, velocity image from forward pass
- Produces:
  - `PostProcessMode` enum: `Native`, `TAA`, `Upscale`, `FrameInterp`
  - `PostProcessPipeline::init(Renderer&)` — creates history buffers (ping-pong pair)
  - `PostProcessPipeline::execute(VkCommandBuffer, Renderer&, PostProcessMode, VkImageView colorInput, VkImageView depthInput, VkImageView velocityInput)` — dispatches active passes
  - `PostProcessPipeline::getOutputImageView() -> VkImageView` — resolved output for presentation
  - `PostProcessPipeline::getOutputImage() -> VkImage`
  - `PostProcessPipeline::cleanup(Renderer&)`
  - `TAAPass::init(Renderer&, VkExtent2D)`
  - `TAAPass::execute(VkCommandBuffer, VkImageView colorInput, VkImageView depthInput, VkImageView velocityInput, VkImageView historyInput, VkImageView outputTarget)`
  - `TAAPass::cleanup(Renderer&)`

- [ ] **Step 1: Create PostProcessPipeline.h**

Create `src/PostProcess/PostProcessPipeline.h`:
```cpp
#pragma once
#include "volk.h"
#include "vk_mem_alloc.h"
#include "TAAPass.h"

class Renderer;

enum class PostProcessMode {
    Native,
    TAA,
    Upscale,
    FrameInterp
};

class PostProcessPipeline {
public:
    void init(Renderer &renderer);
    void execute(VkCommandBuffer cmd, Renderer &renderer, PostProcessMode mode,
                 VkImageView colorInput, VkImageView depthInput, VkImageView velocityInput);
    VkImageView getOutputImageView() const;
    VkImage getOutputImage() const;
    void cleanup(Renderer &renderer);

private:
    // History ping-pong
    VkImage       m_historyImages[2]     {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImageView   m_historyImageViews[2] {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VmaAllocation m_historyAllocations[2]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    uint32_t      m_historyIndex{0};

    // Output (resolved color)
    VkImage       m_outputImage     {VK_NULL_HANDLE};
    VkImageView   m_outputImageView {VK_NULL_HANDLE};
    VmaAllocation m_outputAllocation{VK_NULL_HANDLE};

    // Descriptor set for compute passes
    VkDescriptorSetLayout m_descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool      m_descriptorPool     {VK_NULL_HANDLE};
    VkSampler             m_nearestSampler     {VK_NULL_HANDLE};
    VkSampler             m_linearSampler      {VK_NULL_HANDLE};

    TAAPass m_taaPass{};

    void createHistoryBuffers(Renderer &renderer);
    void createOutputBuffer(Renderer &renderer);
    void createSamplers(Renderer &renderer);
};
```

- [ ] **Step 2: Create PostProcessPipeline.cpp**

Create `src/PostProcess/PostProcessPipeline.cpp`:

```cpp
#include "PostProcessPipeline.h"
#include "Renderer/Renderer.h"
#include "Config.h"

void PostProcessPipeline::init(Renderer &renderer) {
    createHistoryBuffers(renderer);
    createOutputBuffer(renderer);
    createSamplers(renderer);
    m_taaPass.init(renderer, renderer.getSwapchainExtent());
}

void PostProcessPipeline::execute(VkCommandBuffer cmd, Renderer &renderer, PostProcessMode mode,
                                   VkImageView colorInput, VkImageView depthInput, VkImageView velocityInput) {
    if (mode == PostProcessMode::Native) {
        // In native mode, the output IS the color input — copy it
        // (handled by the caller blitting colorInput to swapchain)
        return;
    }

    if (mode == PostProcessMode::TAA) {
        uint32_t readHistory  = m_historyIndex;
        uint32_t writeHistory = 1 - m_historyIndex;

        m_taaPass.execute(cmd, colorInput, depthInput, velocityInput,
                          m_historyImageViews[readHistory],
                          m_historyImages[readHistory],
                          m_outputImageView, m_outputImage,
                          m_historyImageViews[writeHistory],
                          m_historyImages[writeHistory],
                          renderer.getSwapchainExtent());

        m_historyIndex = writeHistory;
    }
}

VkImageView PostProcessPipeline::getOutputImageView() const {
    return m_outputImageView;
}

VkImage PostProcessPipeline::getOutputImage() const {
    return m_outputImage;
}

void PostProcessPipeline::createHistoryBuffers(Renderer &renderer) {
    VkExtent2D extent{renderer.getSwapchainExtent()};

    for (int i = 0; i < 2; i++) {
        VkImageCreateInfo imageCreateInfo{
            .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext         = nullptr,
            .flags         = {},
            .imageType     = VK_IMAGE_TYPE_2D,
            .format        = VK_FORMAT_R16G16B16A16_SFLOAT,
            .extent        = {extent.width, extent.height, 1},
            .mipLevels     = 1,
            .arrayLayers   = 1,
            .samples       = VK_SAMPLE_COUNT_1_BIT,
            .tiling        = VK_IMAGE_TILING_OPTIMAL,
            .usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
        };

        VmaAllocationCreateInfo allocationCreateInfo{.usage = VMA_MEMORY_USAGE_AUTO};

        vmaCreateImage(renderer.getAllocator(), &imageCreateInfo, &allocationCreateInfo,
            &m_historyImages[i], &m_historyAllocations[i], nullptr);

        VkImageViewCreateInfo viewInfo{
            .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext            = nullptr,
            .image            = m_historyImages[i],
            .viewType         = VK_IMAGE_VIEW_TYPE_2D,
            .format           = VK_FORMAT_R16G16B16A16_SFLOAT,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        };
        vkCreateImageView(renderer.getDevice(), &viewInfo, nullptr, &m_historyImageViews[i]);
    }
}

void PostProcessPipeline::createOutputBuffer(Renderer &renderer) {
    VkExtent2D extent{renderer.getSwapchainExtent()};

    VkImageCreateInfo imageCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = {},
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = VK_FORMAT_R16G16B16A16_SFLOAT,
        .extent        = {extent.width, extent.height, 1},
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    VmaAllocationCreateInfo allocationCreateInfo{.usage = VMA_MEMORY_USAGE_AUTO};

    vmaCreateImage(renderer.getAllocator(), &imageCreateInfo, &allocationCreateInfo,
        &m_outputImage, &m_outputAllocation, nullptr);

    VkImageViewCreateInfo viewInfo{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext            = nullptr,
        .image            = m_outputImage,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = VK_FORMAT_R16G16B16A16_SFLOAT,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    vkCreateImageView(renderer.getDevice(), &viewInfo, nullptr, &m_outputImageView);
}

void PostProcessPipeline::createSamplers(Renderer &renderer) {
    VkSamplerCreateInfo nearestSamplerInfo{
        .sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
    };
    vkCreateSampler(renderer.getDevice(), &nearestSamplerInfo, nullptr, &m_nearestSampler);

    VkSamplerCreateInfo linearSamplerInfo{
        .sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
    };
    vkCreateSampler(renderer.getDevice(), &linearSamplerInfo, nullptr, &m_linearSampler);
}

void PostProcessPipeline::cleanup(Renderer &renderer) {
    m_taaPass.cleanup(renderer);

    vkDestroySampler(renderer.getDevice(), m_nearestSampler, nullptr);
    vkDestroySampler(renderer.getDevice(), m_linearSampler, nullptr);
    vkDestroyImageView(renderer.getDevice(), m_outputImageView, nullptr);
    vmaDestroyImage(renderer.getAllocator(), m_outputImage, m_outputAllocation);

    for (int i = 0; i < 2; i++) {
        vkDestroyImageView(renderer.getDevice(), m_historyImageViews[i], nullptr);
        vmaDestroyImage(renderer.getAllocator(), m_historyImages[i], m_historyAllocations[i]);
    }
}
```

- [ ] **Step 3: Create TAAPass.h**

Create `src/PostProcess/TAAPass.h`:
```cpp
#pragma once
#include "volk.h"

class Renderer;

class TAAPass {
public:
    void init(Renderer &renderer, VkExtent2D extent);
    void execute(VkCommandBuffer cmd,
                 VkImageView colorInput, VkImageView depthInput, VkImageView velocityInput,
                 VkImageView historyInput, VkImage historyImage,
                 VkImageView outputTarget, VkImage outputImage,
                 VkImageView historyWriteTarget, VkImage historyWriteImage,
                 VkExtent2D extent);
    void cleanup(Renderer &renderer);

private:
    VkPipeline           m_pipeline      {VK_NULL_HANDLE};
    VkPipelineLayout     m_pipelineLayout{VK_NULL_HANDLE};
    VkDescriptorSetLayout m_descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool     m_descriptorPool{VK_NULL_HANDLE};
    VkDescriptorSet      m_descriptorSet {VK_NULL_HANDLE};
    VkSampler            m_nearestSampler{VK_NULL_HANDLE};
    VkSampler            m_linearSampler {VK_NULL_HANDLE};
};
```

- [ ] **Step 4: Create TAAPass.cpp**

Create `src/PostProcess/TAAPass.cpp`. This creates a compute pipeline that reads the current color, velocity, depth, and history textures, performs reprojection + neighborhood clamping + blending, and writes to both the output and the next history buffer.

The compute shader dispatch is `ceil(width/8) x ceil(height/8) x 1` with 8x8 workgroups.

Descriptor layout uses 5 bindings:
- 0: combined image sampler (current color, nearest)
- 1: combined image sampler (velocity, nearest)
- 2: combined image sampler (depth, nearest)
- 3: combined image sampler (history, linear — for reprojection sampling)
- 4: storage image (output)
- 5: storage image (history write)

```cpp
#include "TAAPass.h"
#include "Renderer/Renderer.h"
#include "Config.h"
#include <fstream>
#include <vector>
#include <stdexcept>

void TAAPass::init(Renderer &renderer, VkExtent2D extent) {
    VkDevice device{renderer.getDevice()};

    // Create samplers
    VkSamplerCreateInfo nearestInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST};
    vkCreateSampler(device, &nearestInfo, nullptr, &m_nearestSampler);
    VkSamplerCreateInfo linearInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR};
    vkCreateSampler(device, &linearInfo, nullptr, &m_linearSampler);

    // Descriptor set layout
    VkDescriptorSetLayoutBinding bindings[]{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // color
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // velocity
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // depth
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // history
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // history write
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 6,
        .pBindings    = bindings
    };
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_descriptorSetLayout);

    // Pipeline layout with push constant for blend factor
    VkPushConstantRange pushRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(float) // taaBlendFactor
    };
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_descriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushRange
    };
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout);

    // Load compute shader
    std::ifstream file(SHADER_DIR "taa_compute.spv", std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Error: failed to open taa compute shader");
    std::vector<char> code(file.tellg());
    file.seekg(0);
    file.read(code.data(), static_cast<std::streamsize>(code.size()));

    VkShaderModuleCreateInfo moduleInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode    = reinterpret_cast<const uint32_t *>(code.data())
    };
    VkShaderModule shaderModule{};
    vkCreateShaderModule(device, &moduleInfo, nullptr, &shaderModule);

    VkComputePipelineCreateInfo pipelineInfo{
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shaderModule,
            .pName  = "main"
        },
        .layout = m_pipelineLayout
    };
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    // Descriptor pool & set
    VkDescriptorPoolSize poolSizes[]{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}
    };
    VkDescriptorPoolCreateInfo poolInfo{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets       = 1,
        .poolSizeCount = 2,
        .pPoolSizes    = poolSizes
    };
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool);

    VkDescriptorSetAllocateInfo allocInfo{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = m_descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &m_descriptorSetLayout
    };
    vkAllocateDescriptorSets(device, &allocInfo, &m_descriptorSet);
}

void TAAPass::execute(VkCommandBuffer cmd,
                       VkImageView colorInput, VkImageView depthInput, VkImageView velocityInput,
                       VkImageView historyInput, VkImage historyImage,
                       VkImageView outputTarget, VkImage outputImage,
                       VkImageView historyWriteTarget, VkImage historyWriteImage,
                       VkExtent2D extent) {
    // Update descriptor set with current frame's images
    VkDescriptorImageInfo colorInfo{m_nearestSampler, colorInput, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo velocityInfo{m_nearestSampler, velocityInput, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo depthInfo{m_nearestSampler, depthInput, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo historyInfo{m_linearSampler, historyInput, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo outputInfo{VK_NULL_HANDLE, outputTarget, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo historyWriteInfo{VK_NULL_HANDLE, historyWriteTarget, VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet writes[]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &colorInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &velocityInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &historyInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descriptorSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descriptorSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &historyWriteInfo, nullptr, nullptr},
    };
    vkUpdateDescriptorSets(VK_NULL_HANDLE, 6, writes, 0, nullptr); // NOTE: device handle needed — pass via renderer or store

    float blendFactor{Config::taaBlendFactor};
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &blendFactor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

    uint32_t groupsX{(extent.width  + 7) / 8};
    uint32_t groupsY{(extent.height + 7) / 8};
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
}

void TAAPass::cleanup(Renderer &renderer) {
    VkDevice device{renderer.getDevice()};
    vkDestroyPipeline(device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
    vkDestroySampler(device, m_nearestSampler, nullptr);
    vkDestroySampler(device, m_linearSampler, nullptr);
}
```

**Important implementation note:** The `vkUpdateDescriptorSets` call above uses `VK_NULL_HANDLE` for the device — this is a placeholder. The actual implementation must store the `VkDevice` handle or pass `Renderer&` to `execute()`. Fix this when implementing.

- [ ] **Step 5: Create taa.slang**

Create `shaders/taa.slang`:
```hlsl
[[vk::binding(0, 0)]] Sampler2D colorTex;
[[vk::binding(1, 0)]] Sampler2D velocityTex;
[[vk::binding(2, 0)]] Sampler2D depthTex;
[[vk::binding(3, 0)]] Sampler2D historyTex;
[[vk::binding(4, 0)]] RWTexture2D<float4> outputTex;
[[vk::binding(5, 0)]] RWTexture2D<float4> historyWriteTex;

[[vk::push_constant]]
struct {
    float blendFactor;
} pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    int2 pixelCoord = int2(dispatchThreadID.xy);
    uint2 texSize;
    outputTex.GetDimensions(texSize.x, texSize.y);

    if (pixelCoord.x >= texSize.x || pixelCoord.y >= texSize.y)
        return;

    float2 uv = (float2(pixelCoord) + 0.5) / float2(texSize);

    // Read current frame data
    float4 currentColor = colorTex.SampleLevel(uv, 0);
    float2 velocity = velocityTex.SampleLevel(uv, 0).rg;

    // Reproject: where was this pixel last frame?
    float2 historyUV = uv - velocity;

    // Reject if reprojected UV is out of bounds
    bool validHistory = all(historyUV >= 0.0) && all(historyUV <= 1.0);

    float4 historyColor = historyTex.SampleLevel(historyUV, 0);

    // Neighborhood clamping (3x3 min/max)
    float3 neighborMin = currentColor.rgb;
    float3 neighborMax = currentColor.rgb;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float2 neighborUV = (float2(pixelCoord + int2(dx, dy)) + 0.5) / float2(texSize);
            float3 neighborColor = colorTex.SampleLevel(neighborUV, 0).rgb;
            neighborMin = min(neighborMin, neighborColor);
            neighborMax = max(neighborMax, neighborColor);
        }
    }

    // Clamp history to neighborhood bounds
    historyColor.rgb = clamp(historyColor.rgb, neighborMin, neighborMax);

    // Blend
    float4 resolved;
    if (validHistory) {
        resolved = float4(lerp(historyColor.rgb, currentColor.rgb, pc.blendFactor), 1.0);
    } else {
        resolved = currentColor;
    }

    outputTex[pixelCoord] = resolved;
    historyWriteTex[pixelCoord] = resolved;
}
```

- [ ] **Step 6: Add shader compilation to CMakeLists.txt**

```cmake
compile_slang_shader(taa main compute)

# Update compile_shaders target:
add_custom_target(compile_shaders DEPENDS
    ${SHADER_OUTPUT_DIR}/forward_vertex.spv
    ${SHADER_OUTPUT_DIR}/forward_fragment.spv
    ${SHADER_OUTPUT_DIR}/taa_compute.spv
)
```

Add PostProcess sources:
```cmake
set(SOURCE
    # ... existing ...
    src/PostProcess/PostProcessPipeline.cpp
    src/PostProcess/TAAPass.cpp
)
```

- [ ] **Step 7: Modify Renderer to render to offscreen target**

The forward pass must render to an offscreen color image (not the swapchain) so the post-process pipeline can read it. Add to Renderer:
- An offscreen color image + image view (`VK_FORMAT_R16G16B16A16_SFLOAT`, same extent as swapchain)
- `getColorImageView()`, `getColorImage()`, `getDepthImageView()`, `getDepthImage()`
- Modify `drawFrame` to render to the offscreen target, then after post-processing blit the output to the swapchain

- [ ] **Step 8: Verify — TAA reduces aliasing**

With TAA enabled, edges should appear smoother compared to Native mode. Moving the camera should NOT produce significant ghosting (neighborhood clamping prevents it). Stationary scenes should show temporally stable, clean edges.

To verify visually: compare Native mode (raw forward pass, slight jitter shimmer visible) with TAA mode (stable, anti-aliased edges).

- [ ] **Step 9: Commit**

```bash
git add src/PostProcess/ shaders/taa.slang src/Renderer/ CMakeLists.txt src/App/
git commit -m "feat: add TAA compute pass with neighborhood clamping and history reprojection"
```

---

### Task 8: Temporal Upscaling Pass

**Files:**
- Create: `src/PostProcess/UpscalePass.h`
- Create: `src/PostProcess/UpscalePass.cpp`
- Create: `src/PostProcess/SharpenPass.h`
- Create: `src/PostProcess/SharpenPass.cpp`
- Create: `shaders/upscale.slang`
- Create: `shaders/sharpen.slang`
- Modify: `src/PostProcess/PostProcessPipeline.h` — add UpscalePass and SharpenPass members
- Modify: `src/PostProcess/PostProcessPipeline.cpp` — handle Upscale mode
- Modify: `src/Renderer/Renderer.cpp` — render at internal resolution when upscaling
- Modify: `CMakeLists.txt` — add sources and shader compilation

**Interfaces:**
- Consumes: Low-res color/depth/velocity from forward pass, full-res history buffers
- Produces:
  - `UpscalePass::init(Renderer&, VkExtent2D internalExtent, VkExtent2D outputExtent)`
  - `UpscalePass::execute(VkCommandBuffer, ...)` — reconstructs full-res from low-res input + history
  - `SharpenPass::init(Renderer&, VkExtent2D)`
  - `SharpenPass::execute(VkCommandBuffer, VkImageView input, VkImageView output, VkExtent2D)` — RCAS sharpening

- [ ] **Step 1: Create UpscalePass.h and UpscalePass.cpp**

Same structure as TAAPass but the compute shader samples the low-res input and writes to the full-res output. The descriptor layout is identical to TAA (color, velocity, depth, history, output, history write).

The key difference in the shader: UV coordinates are scaled by the upscale ratio when sampling the low-res inputs.

- [ ] **Step 2: Create upscale.slang**

Create `shaders/upscale.slang`:
```hlsl
[[vk::binding(0, 0)]] Sampler2D colorTex;      // low-res
[[vk::binding(1, 0)]] Sampler2D velocityTex;    // low-res
[[vk::binding(2, 0)]] Sampler2D depthTex;       // low-res
[[vk::binding(3, 0)]] Sampler2D historyTex;     // full-res
[[vk::binding(4, 0)]] RWTexture2D<float4> outputTex;      // full-res
[[vk::binding(5, 0)]] RWTexture2D<float4> historyWriteTex; // full-res

[[vk::push_constant]]
struct {
    float blendFactor;
    float upscaleRatio;
} pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    int2 outputCoord = int2(dispatchThreadID.xy);
    uint2 outputSize;
    outputTex.GetDimensions(outputSize.x, outputSize.y);

    if (outputCoord.x >= outputSize.x || outputCoord.y >= outputSize.y)
        return;

    float2 outputUV = (float2(outputCoord) + 0.5) / float2(outputSize);

    // Sample low-res inputs at the corresponding position
    float4 currentColor = colorTex.SampleLevel(outputUV, 0);
    float2 velocity = velocityTex.SampleLevel(outputUV, 0).rg;
    float currentDepth = depthTex.SampleLevel(outputUV, 0).r;

    // Reproject into history
    float2 historyUV = outputUV - velocity;
    bool validHistory = all(historyUV >= 0.0) && all(historyUV <= 1.0);

    float4 historyColor = historyTex.SampleLevel(historyUV, 0);

    // Disocclusion detection: check depth discontinuity
    // Compare current depth with history depth (stored in alpha of history)
    float historyDepth = historyColor.a;
    float depthDiff = abs(currentDepth - historyDepth);
    bool disoccluded = depthDiff > 0.01 || !validHistory;

    // Neighborhood clamping on low-res grid
    uint2 inputSize = uint2(float2(outputSize) * pc.upscaleRatio);
    int2 inputCoord = int2(float2(outputCoord) * pc.upscaleRatio);

    float3 neighborMin = currentColor.rgb;
    float3 neighborMax = currentColor.rgb;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float2 neighborUV = (float2(inputCoord + int2(dx, dy)) + 0.5) / float2(inputSize);
            float3 nc = colorTex.SampleLevel(neighborUV, 0).rgb;
            neighborMin = min(neighborMin, nc);
            neighborMax = max(neighborMax, nc);
        }
    }
    historyColor.rgb = clamp(historyColor.rgb, neighborMin, neighborMax);

    // Blend — weight current frame more heavily on disocclusion
    float blend = disoccluded ? 1.0 : pc.blendFactor;
    float4 resolved = float4(lerp(historyColor.rgb, currentColor.rgb, blend), currentDepth);

    outputTex[outputCoord] = resolved;
    historyWriteTex[outputCoord] = resolved;
}
```

- [ ] **Step 3: Create SharpenPass.h, SharpenPass.cpp, and sharpen.slang**

Create `shaders/sharpen.slang` — Contrast Adaptive Sharpening (RCAS-style):
```hlsl
[[vk::binding(0, 0)]] Sampler2D inputTex;
[[vk::binding(1, 0)]] RWTexture2D<float4> outputTex;

[[vk::push_constant]]
struct {
    float sharpenStrength;
} pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    int2 coord = int2(dispatchThreadID.xy);
    uint2 texSize;
    outputTex.GetDimensions(texSize.x, texSize.y);

    if (coord.x >= texSize.x || coord.y >= texSize.y)
        return;

    float2 uv = (float2(coord) + 0.5) / float2(texSize);
    float2 texelSize = 1.0 / float2(texSize);

    // Sample cross pattern
    float3 center = inputTex.SampleLevel(uv, 0).rgb;
    float3 north  = inputTex.SampleLevel(uv + float2(0, -texelSize.y), 0).rgb;
    float3 south  = inputTex.SampleLevel(uv + float2(0,  texelSize.y), 0).rgb;
    float3 west   = inputTex.SampleLevel(uv + float2(-texelSize.x, 0), 0).rgb;
    float3 east   = inputTex.SampleLevel(uv + float2( texelSize.x, 0), 0).rgb;

    // Compute local contrast
    float3 minColor = min(center, min(min(north, south), min(west, east)));
    float3 maxColor = max(center, max(max(north, south), max(west, east)));

    // RCAS weight: higher contrast = less sharpening (adaptive)
    float3 range = maxColor - minColor;
    float3 rcpRange = 1.0 / (range + 0.001);

    // Sharpen: subtract neighbors, amplify center
    float3 sharpened = center + (4.0 * center - north - south - west - east) * pc.sharpenStrength * min(rcpRange, 1.0);

    outputTex[coord] = float4(clamp(sharpened, 0.0, 1.0), 1.0);
}
```

SharpenPass.h/cpp follow the same pattern as TAAPass: compute pipeline with 2 bindings (input sampler, output storage image), push constant for strength.

- [ ] **Step 4: Modify Renderer for internal resolution rendering**

When upscale mode is active, the forward pass should render at `width * upscaleRatio x height * upscaleRatio`. The offscreen color, depth, and velocity targets need to be created at this reduced size. Add a method:
```cpp
void Renderer::setInternalResolution(float ratio);
```

Or create two sets of offscreen targets (full-res and internal-res) and select based on mode.

- [ ] **Step 5: Update PostProcessPipeline for Upscale mode**

In `PostProcessPipeline::execute()`, when mode is `Upscale`:
1. Run UpscalePass (low-res → full-res)
2. Run SharpenPass (full-res → full-res output)

- [ ] **Step 6: Update CMakeLists.txt**

Add sources and shader compilation commands for upscale and sharpen.

- [ ] **Step 7: Verify — upscaled image matches native quality**

Compare Native (full-res) with Upscale mode at 67% internal resolution. The upscaled image should be close to native quality with minor softness (compensated by sharpening). Moving the camera should not produce significant ghosting. Disoccluded areas (newly revealed geometry) should snap to the current frame without smearing.

- [ ] **Step 8: Commit**

```bash
git add src/PostProcess/UpscalePass.* src/PostProcess/SharpenPass.* shaders/upscale.slang shaders/sharpen.slang src/Renderer/ src/PostProcess/PostProcessPipeline.* CMakeLists.txt
git commit -m "feat: add temporal upscaling with disocclusion detection and RCAS sharpening"
```

---

### Task 9: Frame Interpolation

**Files:**
- Create: `src/PostProcess/FrameInterp.h`
- Create: `src/PostProcess/FrameInterp.cpp`
- Create: `shaders/frame_interp.slang`
- Modify: `src/PostProcess/PostProcessPipeline.h` — add FrameInterp member, store previous frame
- Modify: `src/PostProcess/PostProcessPipeline.cpp` — handle FrameInterp mode
- Modify: `src/Renderer/Renderer.cpp` — present interpolated frame between real frames
- Modify: `CMakeLists.txt` — add sources and shader compilation

**Interfaces:**
- Consumes: Two consecutive resolved frames, motion vectors
- Produces:
  - `FrameInterp::init(Renderer&, VkExtent2D)`
  - `FrameInterp::execute(VkCommandBuffer, VkImageView frameN, VkImageView frameN1, VkImageView velocity, VkImageView output, VkExtent2D)`
  - `FrameInterp::cleanup(Renderer&)`
  - An interpolated intermediate frame presented between real frames

- [ ] **Step 1: Create FrameInterp.h and FrameInterp.cpp**

Same compute pipeline pattern. Descriptor layout:
- 0: frame N (sampler)
- 1: frame N+1 (sampler)
- 2: velocity (sampler)
- 3: depth (sampler)
- 4: output (storage image)

- [ ] **Step 2: Create frame_interp.slang**

Create `shaders/frame_interp.slang`:
```hlsl
[[vk::binding(0, 0)]] Sampler2D frameNTex;
[[vk::binding(1, 0)]] Sampler2D frameN1Tex;
[[vk::binding(2, 0)]] Sampler2D velocityTex;
[[vk::binding(3, 0)]] Sampler2D depthTex;
[[vk::binding(4, 0)]] RWTexture2D<float4> outputTex;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    int2 coord = int2(dispatchThreadID.xy);
    uint2 texSize;
    outputTex.GetDimensions(texSize.x, texSize.y);

    if (coord.x >= texSize.x || coord.y >= texSize.y)
        return;

    float2 uv = (float2(coord) + 0.5) / float2(texSize);
    float2 velocity = velocityTex.SampleLevel(uv, 0).rg;

    // Warp frame N forward by half the motion
    float2 warpedUV_forward = uv + velocity * 0.5;
    // Warp frame N+1 backward by half the motion
    float2 warpedUV_backward = uv - velocity * 0.5;

    bool validForward  = all(warpedUV_forward  >= 0.0) && all(warpedUV_forward  <= 1.0);
    bool validBackward = all(warpedUV_backward >= 0.0) && all(warpedUV_backward <= 1.0);

    float4 forwardSample  = frameNTex.SampleLevel(warpedUV_forward, 0);
    float4 backwardSample = frameN1Tex.SampleLevel(warpedUV_backward, 0);

    float4 result;
    if (validForward && validBackward) {
        // Blend the two warped samples
        result = (forwardSample + backwardSample) * 0.5;
    } else if (validForward) {
        result = forwardSample;
    } else if (validBackward) {
        result = backwardSample;
    } else {
        // Fallback: use frame N+1 at current position (no interpolation)
        result = frameN1Tex.SampleLevel(uv, 0);
    }

    // Simple hole-filling: if the result is very different from both neighbors,
    // it's likely a disocclusion hole — use the frame N+1 value
    float4 currentFrame = frameN1Tex.SampleLevel(uv, 0);
    float colorDiff = length(result.rgb - currentFrame.rgb);
    if (colorDiff > 0.5) {
        result = currentFrame;
    }

    outputTex[coord] = result;
}
```

- [ ] **Step 3: Modify PostProcessPipeline for FrameInterp mode**

Store the previous frame's resolved output. After TAA or Upscale produces the current resolved frame, frame interpolation generates a midpoint frame between the stored previous and the current. The pipeline needs to present the interpolated frame first, then the real frame — effectively doubling the apparent frame rate.

Add a member:
```cpp
VkImage       m_previousResolvedImage    {VK_NULL_HANDLE};
VkImageView   m_previousResolvedImageView{VK_NULL_HANDLE};
VmaAllocation m_previousResolvedAllocation{VK_NULL_HANDLE};
```

After each frame, copy the current resolved output to `m_previousResolvedImage`.

- [ ] **Step 4: Modify Renderer for double presentation**

When FrameInterp is active, the frame loop presents two images per drawFrame call:
1. The interpolated frame (generated by compute)
2. The real frame

This requires careful synchronization — the interpolated frame must be blitted to the swapchain and presented, then the real frame acquired and presented. Alternatively, present only the interpolated frame and let the next loop iteration present the real frame (simpler but adds one frame of latency).

- [ ] **Step 5: Update CMakeLists.txt**

Add FrameInterp sources and shader compilation.

- [ ] **Step 6: Verify — interpolated frames appear smooth**

With frame interpolation enabled on top of TAA or Upscale: moving the camera should appear noticeably smoother (effectively 2x frame rate). Check for artifacts at object edges and in disoccluded regions. The interpolated frame should look plausible — not identical to the real frames, but without obvious tearing or warping artifacts.

- [ ] **Step 7: Commit**

```bash
git add src/PostProcess/FrameInterp.* shaders/frame_interp.slang src/PostProcess/PostProcessPipeline.* src/Renderer/ CMakeLists.txt
git commit -m "feat: add frame interpolation via motion-vector-based warping"
```

---

### Task 10: Debug UI & GPU Profiler

**Files:**
- Create: `src/Debug/DebugUI.h`
- Create: `src/Debug/DebugUI.cpp`
- Create: `src/Debug/Profiler.h`
- Create: `src/Debug/Profiler.cpp`
- Modify: `src/App/App.h` — add DebugUI and Profiler members
- Modify: `src/App/App.cpp` — integrate ImGui, toggle cursor capture, pass mode to pipeline
- Modify: `src/Renderer/Renderer.h` — add ImGui descriptor pool
- Modify: `src/Renderer/Renderer.cpp` — initialize ImGui Vulkan backend
- Modify: `CMakeLists.txt` — add Debug sources

**Interfaces:**
- Consumes: `Renderer` (device, instance, queue, swapchain format), `PostProcessMode`, `Profiler` timing data
- Produces:
  - `DebugUI::init(Renderer&, Window&)` — sets up ImGui Vulkan + GLFW backends
  - `DebugUI::beginFrame()` — calls ImGui_ImplVulkan_NewFrame, ImGui_ImplGlfw_NewFrame
  - `DebugUI::render(PostProcessMode&, Profiler&)` — draws the debug panel, returns selected mode
  - `DebugUI::recordCommands(VkCommandBuffer)` — records ImGui draw into command buffer
  - `DebugUI::cleanup()`
  - `Profiler::init(Renderer&, uint32_t numTimestamps)` — creates query pool
  - `Profiler::beginFrame(VkCommandBuffer)` — resets query pool
  - `Profiler::writeTimestamp(VkCommandBuffer, uint32_t index, VkPipelineStageFlagBits2)` — writes timestamp
  - `Profiler::endFrame()` — reads back results
  - `Profiler::getPassTimeMs(uint32_t startIndex, uint32_t endIndex) -> float`
  - `Profiler::cleanup(Renderer&)`

- [ ] **Step 1: Create Profiler.h**

Create `src/Debug/Profiler.h`:
```cpp
#pragma once
#include "volk.h"
#include <vector>
#include <cstdint>

class Renderer;

class Profiler {
public:
    void init(Renderer &renderer, uint32_t numTimestamps);
    void beginFrame(VkCommandBuffer cmd);
    void writeTimestamp(VkCommandBuffer cmd, uint32_t index, VkPipelineStageFlagBits2 stage);
    void endFrame(Renderer &renderer);
    float getPassTimeMs(uint32_t startIndex, uint32_t endIndex) const;
    void cleanup(Renderer &renderer);

private:
    VkQueryPool           m_queryPool       {VK_NULL_HANDLE};
    uint32_t              m_numTimestamps   {0};
    std::vector<uint64_t> m_timestamps      {};
    float                 m_timestampPeriod {0.0f};
    bool                  m_hasResults      {false};
};
```

- [ ] **Step 2: Create Profiler.cpp**

Create `src/Debug/Profiler.cpp`:
```cpp
#include "Profiler.h"
#include "Renderer/Renderer.h"

void Profiler::init(Renderer &renderer, uint32_t numTimestamps) {
    m_numTimestamps = numTimestamps;
    m_timestamps.resize(numTimestamps, 0);

    VkQueryPoolCreateInfo queryPoolInfo{
        .sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType  = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = numTimestamps
    };
    vkCreateQueryPool(renderer.getDevice(), &queryPoolInfo, nullptr, &m_queryPool);

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(renderer.getPhysicalDevice(), &props);
    m_timestampPeriod = props.limits.timestampPeriod;
}

void Profiler::beginFrame(VkCommandBuffer cmd) {
    vkCmdResetQueryPool(cmd, m_queryPool, 0, m_numTimestamps);
    m_hasResults = false;
}

void Profiler::writeTimestamp(VkCommandBuffer cmd, uint32_t index, VkPipelineStageFlagBits2 stage) {
    vkCmdWriteTimestamp2(cmd, stage, m_queryPool, index);
}

void Profiler::endFrame(Renderer &renderer) {
    VkResult result = vkGetQueryPoolResults(
        renderer.getDevice(), m_queryPool,
        0, m_numTimestamps,
        m_timestamps.size() * sizeof(uint64_t), m_timestamps.data(),
        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT
    );
    m_hasResults = (result == VK_SUCCESS);
}

float Profiler::getPassTimeMs(uint32_t startIndex, uint32_t endIndex) const {
    if (!m_hasResults || startIndex >= m_numTimestamps || endIndex >= m_numTimestamps)
        return 0.0f;
    uint64_t delta = m_timestamps[endIndex] - m_timestamps[startIndex];
    return static_cast<float>(delta) * m_timestampPeriod / 1e6f; // ns to ms
}

void Profiler::cleanup(Renderer &renderer) {
    vkDestroyQueryPool(renderer.getDevice(), m_queryPool, nullptr);
}
```

- [ ] **Step 3: Create DebugUI.h**

Create `src/Debug/DebugUI.h`:
```cpp
#pragma once
#include "volk.h"

class Renderer;
class Window;
class Profiler;
enum class PostProcessMode;

class DebugUI {
public:
    void init(Renderer &renderer, Window &window);
    void beginFrame();
    void render(PostProcessMode &mode, Profiler &profiler);
    void recordCommands(VkCommandBuffer cmd, VkImageView targetImageView, VkExtent2D extent);
    void cleanup(Renderer &renderer);

private:
    VkDescriptorPool m_imguiPool{VK_NULL_HANDLE};
};
```

- [ ] **Step 4: Create DebugUI.cpp**

Create `src/Debug/DebugUI.cpp`:
```cpp
#include "DebugUI.h"
#include "Profiler.h"
#include "Renderer/Renderer.h"
#include "Window/Window.h"
#include "PostProcess/PostProcessPipeline.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

void DebugUI::init(Renderer &renderer, Window &window) {
    // Create descriptor pool for ImGui
    VkDescriptorPoolSize poolSizes[]{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}
    };
    VkDescriptorPoolCreateInfo poolInfo{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets       = 1,
        .poolSizeCount = 1,
        .pPoolSizes    = poolSizes
    };
    vkCreateDescriptorPool(renderer.getDevice(), &poolInfo, nullptr, &m_imguiPool);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window.getWindow(), true);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance       = VK_NULL_HANDLE; // NOTE: need to expose from Renderer
    initInfo.PhysicalDevice = renderer.getPhysicalDevice();
    initInfo.Device         = renderer.getDevice();
    initInfo.QueueFamily    = renderer.getQueueIndex();
    initInfo.Queue          = renderer.getQueue();
    initInfo.DescriptorPool = m_imguiPool;
    initInfo.MinImageCount  = 2;
    initInfo.ImageCount     = 2;
    initInfo.UseDynamicRendering = true;

    VkFormat colorFormat = renderer.getSwapchainFormat();
    VkPipelineRenderingCreateInfo renderingInfo{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &colorFormat
    };
    initInfo.PipelineRenderingCreateInfo = renderingInfo;

    ImGui_ImplVulkan_Init(&initInfo);
}

void DebugUI::beginFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void DebugUI::render(PostProcessMode &mode, Profiler &profiler) {
    ImGui::Begin("Temporal Upscaling Renderer");

    // Mode selector
    const char *modeNames[]{"Native", "TAA", "Upscale", "Frame Interp"};
    int currentMode = static_cast<int>(mode);
    if (ImGui::Combo("Mode", &currentMode, modeNames, 4)) {
        mode = static_cast<PostProcessMode>(currentMode);
    }

    ImGui::Separator();
    ImGui::Text("GPU Timings:");

    // Timestamp indices: 0=frame start, 1=forward end, 2=postprocess end
    float forwardMs     = profiler.getPassTimeMs(0, 1);
    float postprocessMs = profiler.getPassTimeMs(1, 2);
    float totalMs       = profiler.getPassTimeMs(0, 2);

    ImGui::Text("Forward Pass:  %.2f ms", forwardMs);
    ImGui::Text("Post-Process:  %.2f ms", postprocessMs);
    ImGui::Text("Total Frame:   %.2f ms", totalMs);
    ImGui::Text("FPS:           %.0f", totalMs > 0 ? 1000.0f / totalMs : 0.0f);

    // Simple timing bars
    float maxMs = 16.67f; // 60 FPS reference line
    ImGui::ProgressBar(forwardMs / maxMs, ImVec2(-1, 0), "Forward");
    ImGui::ProgressBar(postprocessMs / maxMs, ImVec2(-1, 0), "PostProcess");

    ImGui::End();
    ImGui::Render();
}

void DebugUI::recordCommands(VkCommandBuffer cmd, VkImageView targetImageView, VkExtent2D extent) {
    VkRenderingAttachmentInfo colorAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = targetImageView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
    };

    VkRenderingInfo renderingInfo{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea           = {{0, 0}, extent},
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachment,
    };

    vkCmdBeginRendering(cmd, &renderingInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
}

void DebugUI::cleanup(Renderer &renderer) {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(renderer.getDevice(), m_imguiPool, nullptr);
}
```

- [ ] **Step 5: Integrate in App**

Add to App:
- `DebugUI m_debugUI{}` and `Profiler m_profiler{}`
- `PostProcessMode m_mode{PostProcessMode::TAA}`
- Toggle cursor capture with a key (e.g., Tab) so the mouse can interact with ImGui
- In the main loop: `m_debugUI.beginFrame()`, `m_debugUI.render(m_mode, m_profiler)`, pass mode to pipeline, record ImGui commands after post-process

- [ ] **Step 6: Update CMakeLists.txt**

Add Debug sources.

- [ ] **Step 7: Verify — ImGui panel visible and functional**

Run the application. Press Tab to release the cursor. The ImGui panel should be visible with:
- Mode dropdown (Native/TAA/Upscale/Frame Interp)
- GPU timing bars updating in real-time
- Switching modes should visually change the output

- [ ] **Step 8: Commit**

```bash
git add src/Debug/ src/App/ src/Renderer/ CMakeLists.txt
git commit -m "feat: add ImGui debug panel with GPU profiler and mode switching"
```

---

### Task 11: Polish — README, Artifact Documentation

**Files:**
- Modify: `README.md` — comprehensive project description
- Modify: `src/App/App.cpp` — any final integration fixes

**Interfaces:**
- Consumes: everything
- Produces: A polished, presentable project

- [ ] **Step 1: Write README.md**

Write a comprehensive README covering:
- Project overview: what this is and why it exists (temporal upscaling renderer demonstrating FSR/Fluid Motion Frames concepts)
- Screenshots/GIF of the renderer running (placeholder — capture after everything works)
- Architecture overview: the pipeline stages, data flow
- Build instructions (CMake, dependencies)
- Controls (WASD, mouse, Tab for ImGui, mode switching)
- Technical deep-dive section: explain one hard problem you solved (e.g., disocclusion detection, ghosting on fast camera motion, thin geometry flickering)
- Performance numbers: GPU timing for each pass at various internal resolutions
- Comparison: Native vs TAA vs Upscaled (visual quality, frame time)

- [ ] **Step 2: Final integration test**

Run through every mode, verify:
- Native: clean forward render, slight jitter shimmer visible
- TAA: stable anti-aliased edges, no ghosting on camera movement
- Upscale at 67%: near-native quality, fast disocclusion recovery
- Frame Interp: visibly smoother motion, no major artifacts
- ImGui: mode switching works, GPU timings update, cursor toggle works
- No validation errors

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README with architecture overview and build instructions"
```
