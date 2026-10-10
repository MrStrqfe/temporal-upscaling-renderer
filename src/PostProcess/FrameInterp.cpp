#include "FrameInterp.h"
#include "Renderer/Renderer.h"
#include "Config.h"
#include <fstream>
#include <vector>
#include <stdexcept>

void FrameInterp::init(Renderer &renderer, VkExtent2D /*extent*/) {
    m_device = renderer.getDevice();

    // Clamp so warped lookups near the border don't wrap to the opposite edge
    VkSamplerCreateInfo linearInfo{
        .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter    = VK_FILTER_LINEAR,
        .minFilter    = VK_FILTER_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
    };
    vkCreateSampler(m_device, &linearInfo, nullptr, &m_linearSampler);

    // Velocity and depth must not be filtered across silhouettes
    VkSamplerCreateInfo nearestInfo{
        .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter    = VK_FILTER_NEAREST,
        .minFilter    = VK_FILTER_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
    };
    vkCreateSampler(m_device, &nearestInfo, nullptr, &m_nearestSampler);

    // Descriptor set layout: 4 combined image samplers + 1 storage image
    VkDescriptorSetLayoutBinding bindings[]{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // frame N
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // frame N+1
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // velocity
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // depth
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 5,
        .pBindings    = bindings
    };
    vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorSetLayout);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts    = &m_descriptorSetLayout
    };
    vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout);

    // Load compute shader
    std::ifstream file(std::string(SHADER_DIR) + "frame_interp_compute.spv", std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Error: failed to open frame interpolation compute shader");
    std::vector<char> code(file.tellg());
    file.seekg(0);
    file.read(code.data(), static_cast<std::streamsize>(code.size()));

    VkShaderModuleCreateInfo moduleInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode    = reinterpret_cast<const uint32_t *>(code.data())
    };
    VkShaderModule shaderModule{};
    vkCreateShaderModule(m_device, &moduleInfo, nullptr, &shaderModule);

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
    vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline);
    vkDestroyShaderModule(m_device, shaderModule, nullptr);

    // Descriptor pool & sets — one per frame in flight
    VkDescriptorPoolSize poolSizes[]{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 * Config::maxFramesInFlight},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1 * Config::maxFramesInFlight}
    };
    VkDescriptorPoolCreateInfo poolInfo{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = static_cast<uint32_t>(Config::maxFramesInFlight),
        .poolSizeCount = 2,
        .pPoolSizes    = poolSizes
    };
    vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool);

    VkDescriptorSetLayout layouts[2]{m_descriptorSetLayout, m_descriptorSetLayout};
    VkDescriptorSetAllocateInfo allocInfo{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = m_descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(Config::maxFramesInFlight),
        .pSetLayouts        = layouts
    };
    vkAllocateDescriptorSets(m_device, &allocInfo, m_descriptorSets);
}

void FrameInterp::execute(VkCommandBuffer cmd,
                           VkImageView frameNView, VkImage frameNImage,
                           VkImageView frameN1View,  VkImage frameN1Image,
                           VkImageView velocityView, VkImageView depthView,
                           VkImageView outputView,   VkImage outputImage,
                           VkExtent2D extent) {
    // Frame N: copied in at the end of the previous frame → sampled read
    // Frame N+1: written by the sharpen pass → sampled read (stays GENERAL; it is blitted next)
    // Output: UNDEFINED → GENERAL for storage write
    VkImageMemoryBarrier2 barriers[]{
        {
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = VK_PIPELINE_STAGE_2_COPY_BIT,
            .srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask       = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            .oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = frameNImage,
            .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        },
        {
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask       = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask       = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            .oldLayout           = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = frameN1Image,
            .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        },
        {
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = VK_PIPELINE_STAGE_2_BLIT_BIT,
            .srcAccessMask       = VK_ACCESS_2_NONE,
            .dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask       = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = outputImage,
            .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        }
    };
    VkDependencyInfo depInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 3,
        .pImageMemoryBarriers    = barriers
    };
    vkCmdPipelineBarrier2(cmd, &depInfo);

    // Update descriptor set for this frame
    VkDescriptorSet currentSet{m_descriptorSets[m_frameIndex]};

    VkDescriptorImageInfo frameNInfo  {m_linearSampler,  frameNView,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo frameN1Info {m_linearSampler,  frameN1View,  VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo velocityInfo{m_nearestSampler, velocityView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo depthInfo   {m_nearestSampler, depthView,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo outputInfo  {VK_NULL_HANDLE,   outputView,   VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet writes[]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, currentSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &frameNInfo,   nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, currentSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &frameN1Info,  nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, currentSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &velocityInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, currentSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo,    nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, currentSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &outputInfo,   nullptr, nullptr},
    };
    vkUpdateDescriptorSets(m_device, 5, writes, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &currentSet, 0, nullptr);

    uint32_t groupsX{(extent.width  + 7) / 8};
    uint32_t groupsY{(extent.height + 7) / 8};
    vkCmdDispatch(cmd, groupsX, groupsY, 1);

    m_frameIndex = (m_frameIndex + 1) % Config::maxFramesInFlight;
}

void FrameInterp::cleanup(Renderer &/*renderer*/) {
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
    vkDestroySampler(m_device, m_linearSampler, nullptr);
    vkDestroySampler(m_device, m_nearestSampler, nullptr);
}
