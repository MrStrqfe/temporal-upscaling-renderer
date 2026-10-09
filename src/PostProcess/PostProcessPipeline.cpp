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
        // In native mode, the output IS the color input — caller handles blit
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
