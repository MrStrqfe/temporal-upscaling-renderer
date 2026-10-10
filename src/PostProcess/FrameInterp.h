#pragma once
#include "volk.h"

class Renderer;

class FrameInterp {
public:
    void init(Renderer &renderer, VkExtent2D extent);
    // frameN must be in TRANSFER_DST_OPTIMAL (last written by a copy),
    // frameN1 in GENERAL (last written by compute), velocity/depth in SHADER_READ_ONLY_OPTIMAL.
    // Leaves output in GENERAL.
    void execute(VkCommandBuffer cmd,
                 VkImageView frameNView, VkImage frameNImage,
                 VkImageView frameN1View,  VkImage frameN1Image,
                 VkImageView velocityView, VkImageView depthView,
                 VkImageView outputView,   VkImage outputImage,
                 VkExtent2D extent);
    void cleanup(Renderer &renderer);

private:
    VkDevice              m_device             {VK_NULL_HANDLE};
    VkPipeline            m_pipeline           {VK_NULL_HANDLE};
    VkPipelineLayout      m_pipelineLayout     {VK_NULL_HANDLE};
    VkDescriptorSetLayout m_descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool      m_descriptorPool     {VK_NULL_HANDLE};
    VkSampler             m_linearSampler      {VK_NULL_HANDLE};
    VkSampler             m_nearestSampler     {VK_NULL_HANDLE};
    VkDescriptorSet       m_descriptorSets[2]  {VK_NULL_HANDLE, VK_NULL_HANDLE};
    uint32_t              m_frameIndex         {0};
};
