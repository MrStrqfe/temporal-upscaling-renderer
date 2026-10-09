cmake_minimum_required(VERSION 3.24)

project(temporal-upscaling-renderer)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# packages
find_package(glfw3 3.3 REQUIRED)
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

FetchContent_Declare(
    vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.1.0
)
FetchContent_MakeAvailable(vma)

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

# Compile all shaders — only lists .spv outputs when slangc is available
if (SLANGC_PATH)
    add_custom_target(compile_shaders DEPENDS
        ${SHADER_OUTPUT_DIR}/forward_vertex.spv
        ${SHADER_OUTPUT_DIR}/forward_fragment.spv
    )
else()
    add_custom_target(compile_shaders)
endif()

# source files — starts minimal; later tasks add their own sources
set(SOURCE
    src/main.cpp
    src/Window/Window.cpp
    src/Input/Input.cpp
    src/Camera/Camera.cpp
    src/Renderer/Renderer.cpp
    src/ModelLoader/ModelLoader.cpp
    src/App/App.cpp
)

set(LIBS
    libs/volk.c
    libs/vmaImplementation.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE} ${LIBS} ${IMGUI_SOURCES})
add_dependencies(${PROJECT_NAME} compile_shaders)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)

# Vulkan / MoltenVK include paths
set(MOLTENVK_PREFIX "/opt/homebrew/opt/molten-vk")
set(VULKAN_HEADERS_INCLUDE "/opt/homebrew/include")

target_include_directories(${PROJECT_NAME} PRIVATE
    src
    libs
    models
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${vma_SOURCE_DIR}/include
    ${VULKAN_HEADERS_INCLUDE}
    ${MOLTENVK_PREFIX}/include
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
    assimp::assimp
    glm::glm
)

# Link MoltenVK for runtime Vulkan ICD (volk will dlopen it)
# On macOS with MoltenVK, we link the dylib so volk can find it via RPATH
target_link_directories(${PROJECT_NAME} PRIVATE
    ${MOLTENVK_PREFIX}/lib
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    MoltenVK
)
