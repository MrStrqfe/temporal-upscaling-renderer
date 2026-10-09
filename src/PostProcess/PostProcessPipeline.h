#pragma once
#include "volk.h"
#include "vk_mem_alloc.h"
#include "TAAPass.h"
#include "UpscalePass.h"
#include "SharpenPass.h"

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

    // Output (resolved color — used by TAA and as intermediate for Upscale)
    VkImage       m_outputImage     {VK_NULL_HANDLE};
    VkImageView   m_outputImageView {VK_NULL_HANDLE};
    VmaAllocation m_outputAllocation{VK_NULL_HANDLE};

    // Final sharpen output (Upscale mode only — SharpenPass writes here)
    VkImage       m_sharpenOutputImage     {VK_NULL_HANDLE};
    VkImageView   m_sharpenOutputImageView {VK_NULL_HANDLE};
    VmaAllocation m_sharpenOutputAllocation{VK_NULL_HANDLE};

    // Tracks which image is the final output this frame (set during execute)
    VkImage       m_finalOutputImage    {VK_NULL_HANDLE};
    VkImageView   m_finalOutputImageView{VK_NULL_HANDLE};

    // Samplers
    VkSampler             m_nearestSampler     {VK_NULL_HANDLE};
    VkSampler             m_linearSampler      {VK_NULL_HANDLE};

    TAAPass       m_taaPass{};
    UpscalePass   m_upscalePass{};
    SharpenPass   m_sharpenPass{};

    void createHistoryBuffers(Renderer &renderer);
    void createOutputBuffer(Renderer &renderer);
    void createSharpenOutputBuffer(Renderer &renderer);
    void createSamplers(Renderer &renderer);
};
