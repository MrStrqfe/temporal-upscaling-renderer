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
