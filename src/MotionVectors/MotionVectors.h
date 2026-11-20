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
