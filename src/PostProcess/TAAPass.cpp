#include "TAAPass.h"
#include "Renderer/Renderer.h"
#include "Config.h"
#include <fstream>
#include <vector>
#include <stdexcept>

void TAAPass::init(Renderer &renderer, VkExtent2D extent) {
    m_device = renderer.getDevice();

    // Create samplers
    VkSamplerCreateInfo nearestInfo{
        .sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST
    };
    vkCreateSampler(m_device, &nearestInfo, nullptr, &m_nearestSampler);

    VkSamplerCreateInfo linearInfo{
        .sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR
    };
    vkCreateSampler(m_device, &linearInfo, nullptr, &m_linearSampler);

    // Descriptor set layout: 4 combined image samplers + 2 storage images
    VkDescriptorSetLayoutBinding bindings[]{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // color
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // velocity
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // depth
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // history
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // history write
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 6,
        .pBindings    = bindings
    };
    vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorSetLayout);

    // Pipeline layout with push constant for blend factor
    VkPushConstantRange pushRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(float) // taaBlendFactor
    };
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_descriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushRange
    };
    vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout);

    // Load compute shader
    std::ifstream file(std::string(SHADER_DIR) + "taa_compute.spv", std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Error: failed to open taa compute shader");
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

    // Descriptor pool & set -- no UPDATE_AFTER_BIND needed since we update before binding
    VkDescriptorPoolSize poolSizes[]{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}
    };
    VkDescriptorPoolCreateInfo poolInfo{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = 1,
        .poolSizeCount = 2,
        .pPoolSizes    = poolSizes
    };
    vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool);

    VkDescriptorSetAllocateInfo allocInfo{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = m_descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &m_descriptorSetLayout
    };
    vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSet);
}

void TAAPass::execute(VkCommandBuffer cmd,
                       VkImageView colorInput, VkImageView depthInput, VkImageView velocityInput,
                       VkImageView historyInput, VkImage historyImage,
                       VkImageView outputTarget, VkImage outputImage,
                       VkImageView historyWriteTarget, VkImage historyWriteImage,
                       VkExtent2D extent) {
    // Transition history read to GENERAL, output and history write to GENERAL
    VkImageMemoryBarrier2 barriers[]{
        {
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask       = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask       = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = historyImage,
            .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        },
        {
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
            .srcAccessMask       = VK_ACCESS_2_NONE,
            .dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask       = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = outputImage,
            .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        },
        {
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
            .srcAccessMask       = VK_ACCESS_2_NONE,
            .dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask       = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = historyWriteImage,
            .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        }
    };
    VkDependencyInfo depInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 3,
        .pImageMemoryBarriers    = barriers
    };
    vkCmdPipelineBarrier2(cmd, &depInfo);

    // Update descriptor set with current frame's images
    VkDescriptorImageInfo colorInfo{m_nearestSampler, colorInput, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo velocityInfo{m_nearestSampler, velocityInput, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo depthInfo{m_nearestSampler, depthInput, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo historyInfo{m_linearSampler, historyInput, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo outputInfo{VK_NULL_HANDLE, outputTarget, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo historyWriteInfo{VK_NULL_HANDLE, historyWriteTarget, VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet writes[]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &colorInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &velocityInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &historyInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descriptorSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descriptorSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &historyWriteInfo, nullptr, nullptr},
    };
    vkUpdateDescriptorSets(m_device, 6, writes, 0, nullptr);

    float blendFactor{Config::taaBlendFactor};
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &blendFactor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

    uint32_t groupsX{(extent.width  + 7) / 8};
    uint32_t groupsY{(extent.height + 7) / 8};
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
}

void TAAPass::cleanup(Renderer &renderer) {
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
    vkDestroySampler(m_device, m_nearestSampler, nullptr);
    vkDestroySampler(m_device, m_linearSampler, nullptr);
}
