#include "PostProcessPipeline.h"
#include "Renderer/Renderer.h"
#include "Config.h"

void PostProcessPipeline::init(Renderer &renderer) {
    createHistoryBuffers(renderer);
    createOutputBuffer(renderer);
    createSharpenOutputBuffer(renderer);
    createSamplers(renderer);
    m_taaPass.init(renderer, renderer.getSwapchainExtent());

    VkExtent2D swapExtent{renderer.getSwapchainExtent()};
    VkExtent2D internalExtent{
        static_cast<uint32_t>(swapExtent.width  * Config::upscaleRatio),
        static_cast<uint32_t>(swapExtent.height * Config::upscaleRatio)
    };
    m_upscalePass.init(renderer, internalExtent, swapExtent);
    m_sharpenPass.init(renderer, swapExtent);

    createFullResImage(renderer, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                       m_previousFrameImage, m_previousFrameImageView, m_previousFrameAllocation);
    createFullResImage(renderer, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                       m_interpolatedImage, m_interpolatedImageView, m_interpolatedAllocation);
    m_frameInterp.init(renderer, swapExtent);

    // Default final output points to the main output image
    m_finalOutputImage     = m_outputImage;
    m_finalOutputImageView = m_outputImageView;
}

void PostProcessPipeline::execute(VkCommandBuffer cmd, Renderer &renderer, PostProcessMode mode,
                                   VkImageView colorInput, VkImageView depthInput, VkImageView velocityInput) {
    if (mode == PostProcessMode::Native) {
        // In native mode, the output IS the color input — caller handles blit
        m_previousFrameValid = false;
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

        m_finalOutputImage     = m_outputImage;
        m_finalOutputImageView = m_outputImageView;
    }

    // The stored previous frame goes stale as soon as interpolation stops copying into it
    if (mode != PostProcessMode::FrameInterp) {
        m_previousFrameValid = false;
    }

    // Frame interpolation runs on top of the upscale chain
    if (mode == PostProcessMode::Upscale || mode == PostProcessMode::FrameInterp) {
        uint32_t readHistory  = m_historyIndex;
        uint32_t writeHistory = 1 - m_historyIndex;

        VkExtent2D outputExtent{renderer.getSwapchainExtent()};
        VkExtent2D internalExtent{
            static_cast<uint32_t>(outputExtent.width  * Config::upscaleRatio),
            static_cast<uint32_t>(outputExtent.height * Config::upscaleRatio)
        };

        // Step 1: UpscalePass — low-res inputs → full-res output + history write
        m_upscalePass.execute(cmd, colorInput, depthInput, velocityInput,
                              m_historyImageViews[readHistory],
                              m_historyImages[readHistory],
                              m_outputImageView, m_outputImage,
                              m_historyImageViews[writeHistory],
                              m_historyImages[writeHistory],
                              outputExtent);

        m_historyIndex = writeHistory;

        // Step 2: SharpenPass — full-res upscale output → sharpened final output
        // SharpenPass internally transitions m_outputImage GENERAL→SHADER_READ_ONLY_OPTIMAL
        // and m_sharpenOutputImage UNDEFINED→GENERAL
        m_sharpenPass.execute(cmd,
                              m_outputImageView, m_outputImage,
                              m_sharpenOutputImageView, m_sharpenOutputImage,
                              outputExtent);

        m_finalOutputImage     = m_sharpenOutputImage;
        m_finalOutputImageView = m_sharpenOutputImageView;
    }

    if (willInterpolate(mode)) {
        m_frameInterp.execute(cmd,
                              m_previousFrameImageView, m_previousFrameImage,
                              m_finalOutputImageView,   m_finalOutputImage,
                              velocityInput, depthInput,
                              m_interpolatedImageView,  m_interpolatedImage,
                              renderer.getSwapchainExtent());
    }
}

bool PostProcessPipeline::willInterpolate(PostProcessMode mode) const {
    return mode == PostProcessMode::FrameInterp && m_previousFrameValid;
}

VkImage PostProcessPipeline::getInterpolatedImage() const {
    return m_interpolatedImage;
}

void PostProcessPipeline::recordPreviousFrameCopy(VkCommandBuffer cmd, Renderer &renderer) {
    // Previous contents are fully overwritten, so discard them; wait for this frame's
    // interpolation pass to finish reading them first
    VkImageMemoryBarrier2 toTransferDst{
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask       = VK_ACCESS_2_NONE,
        .dstStageMask        = VK_PIPELINE_STAGE_2_COPY_BIT,
        .dstAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = m_previousFrameImage,
        .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    VkDependencyInfo depInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &toTransferDst
    };
    vkCmdPipelineBarrier2(cmd, &depInfo);

    VkExtent2D extent{renderer.getSwapchainExtent()};
    VkImageCopy2 region{
        .sType          = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .extent         = {extent.width, extent.height, 1}
    };
    VkCopyImageInfo2 copyInfo{
        .sType          = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
        .srcImage       = m_finalOutputImage,
        .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .dstImage       = m_previousFrameImage,
        .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .regionCount    = 1,
        .pRegions       = &region
    };
    vkCmdCopyImage2(cmd, &copyInfo);

    m_previousFrameValid = true;
}

VkImageView PostProcessPipeline::getOutputImageView() const {
    return m_finalOutputImageView;
}

VkImage PostProcessPipeline::getOutputImage() const {
    return m_finalOutputImage;
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

void PostProcessPipeline::createSharpenOutputBuffer(Renderer &renderer) {
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
        &m_sharpenOutputImage, &m_sharpenOutputAllocation, nullptr);

    VkImageViewCreateInfo viewInfo{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext            = nullptr,
        .image            = m_sharpenOutputImage,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = VK_FORMAT_R16G16B16A16_SFLOAT,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    vkCreateImageView(renderer.getDevice(), &viewInfo, nullptr, &m_sharpenOutputImageView);
}

void PostProcessPipeline::createFullResImage(Renderer &renderer, VkImageUsageFlags usage,
                                              VkImage &image, VkImageView &view, VmaAllocation &allocation) {
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
        .usage         = usage,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    VmaAllocationCreateInfo allocationCreateInfo{.usage = VMA_MEMORY_USAGE_AUTO};

    vmaCreateImage(renderer.getAllocator(), &imageCreateInfo, &allocationCreateInfo,
        &image, &allocation, nullptr);

    VkImageViewCreateInfo viewInfo{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext            = nullptr,
        .image            = image,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = VK_FORMAT_R16G16B16A16_SFLOAT,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    vkCreateImageView(renderer.getDevice(), &viewInfo, nullptr, &view);
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
    m_upscalePass.cleanup(renderer);
    m_sharpenPass.cleanup(renderer);
    m_frameInterp.cleanup(renderer);

    vkDestroyImageView(renderer.getDevice(), m_interpolatedImageView, nullptr);
    vmaDestroyImage(renderer.getAllocator(), m_interpolatedImage, m_interpolatedAllocation);
    vkDestroyImageView(renderer.getDevice(), m_previousFrameImageView, nullptr);
    vmaDestroyImage(renderer.getAllocator(), m_previousFrameImage, m_previousFrameAllocation);

    vkDestroySampler(renderer.getDevice(), m_nearestSampler, nullptr);
    vkDestroySampler(renderer.getDevice(), m_linearSampler, nullptr);

    vkDestroyImageView(renderer.getDevice(), m_sharpenOutputImageView, nullptr);
    vmaDestroyImage(renderer.getAllocator(), m_sharpenOutputImage, m_sharpenOutputAllocation);

    vkDestroyImageView(renderer.getDevice(), m_outputImageView, nullptr);
    vmaDestroyImage(renderer.getAllocator(), m_outputImage, m_outputAllocation);

    for (int i = 0; i < 2; i++) {
        vkDestroyImageView(renderer.getDevice(), m_historyImageViews[i], nullptr);
        vmaDestroyImage(renderer.getAllocator(), m_historyImages[i], m_historyAllocations[i]);
    }
}
