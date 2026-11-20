#pragma once
#include "volk.h"
#include "vk_mem_alloc.h"
#include "TAAPass.h"
#include "UpscalePass.h"
#include "SharpenPass.h"
#include "FrameInterp.h"

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
    // True when execute() in this mode will also produce an interpolated frame to present first
    bool willInterpolate(PostProcessMode mode) const;
    VkImage getInterpolatedImage() const;
    // Saves this frame's output as frame N for the next interpolation.
    // The output image must already be in TRANSFER_SRC_OPTIMAL (i.e. after the swapchain blit).
    void recordPreviousFrameCopy(VkCommandBuffer cmd, Renderer &renderer);
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

    // Frame interpolation: previous resolved frame (frame N) and the generated midpoint frame
    VkImage       m_previousFrameImage          {VK_NULL_HANDLE};
    VkImageView   m_previousFrameImageView      {VK_NULL_HANDLE};
    VmaAllocation m_previousFrameAllocation     {VK_NULL_HANDLE};
    bool          m_previousFrameValid          {false};
    VkImage       m_interpolatedImage           {VK_NULL_HANDLE};
    VkImageView   m_interpolatedImageView       {VK_NULL_HANDLE};
    VmaAllocation m_interpolatedAllocation      {VK_NULL_HANDLE};

    // Samplers
    VkSampler             m_nearestSampler     {VK_NULL_HANDLE};
    VkSampler             m_linearSampler      {VK_NULL_HANDLE};

    TAAPass       m_taaPass{};
    UpscalePass   m_upscalePass{};
    SharpenPass   m_sharpenPass{};
    FrameInterp   m_frameInterp{};

    void createHistoryBuffers(Renderer &renderer);
    void createOutputBuffer(Renderer &renderer);
    void createSharpenOutputBuffer(Renderer &renderer);
    void createSamplers(Renderer &renderer);
    void createFullResImage(Renderer &renderer, VkImageUsageFlags usage,
                            VkImage &image, VkImageView &view, VmaAllocation &allocation);
};
