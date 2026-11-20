#pragma once
#include "volk.h"

class Renderer;

class SharpenPass {
public:
    void init(Renderer &renderer, VkExtent2D extent);
    void execute(VkCommandBuffer cmd,
                 VkImageView inputView, VkImage inputImage,
                 VkImageView outputView, VkImage outputImage,
                 VkExtent2D extent);
    void cleanup(Renderer &renderer);

private:
    VkDevice              m_device             {VK_NULL_HANDLE};
    VkPipeline            m_pipeline           {VK_NULL_HANDLE};
    VkPipelineLayout      m_pipelineLayout     {VK_NULL_HANDLE};
    VkDescriptorSetLayout m_descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool      m_descriptorPool     {VK_NULL_HANDLE};
    VkSampler             m_linearSampler      {VK_NULL_HANDLE};
    VkDescriptorSet       m_descriptorSets[2]  {VK_NULL_HANDLE, VK_NULL_HANDLE};
    uint32_t              m_frameIndex         {0};
};
