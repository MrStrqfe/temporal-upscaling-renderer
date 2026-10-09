#pragma once
#include "volk.h"
#include "vk_mem_alloc.h"
#include <vector>
#include <cstdint>
#include <string>
class Window;
class Camera;
struct Vertex;
struct Material;

class Renderer {
public:
    void initVulkan(Window &window);
    void drawFrame(Window &window, Camera &camera);
    void uploadModel(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices);
    uint32_t uploadTexture(const std::string &path, VkFormat format);
    void uploadMaterials(std::vector<Material> &materials);
    VkDevice getDevice() const;
    VkPhysicalDevice getPhysicalDevice() const;
    VmaAllocator getAllocator() const;
    VkQueue getQueue() const;
    uint32_t getQueueIndex() const;
    VkCommandPool getCommandPool() const;
    VkExtent2D getSwapchainExtent() const;
    VkFormat getSwapchainFormat() const;
    void setVelocityTarget(VkImageView velocityImageView, VkImage velocityImage);
    void cleanup();

private:
    VkInstance                   m_instance                {VK_NULL_HANDLE};
    VkPhysicalDevice             m_physicalDevice          {VK_NULL_HANDLE};
    VkDevice                     m_device                  {VK_NULL_HANDLE};
    VkQueue                      m_queue                   {VK_NULL_HANDLE};
    uint32_t                     m_queueIndex              {};
    VmaAllocator                 m_allocator               {VK_NULL_HANDLE};
    VkSurfaceKHR                 m_surface                 {VK_NULL_HANDLE};
    // swapchain
    VkSwapchainKHR               m_swapchain               {VK_NULL_HANDLE};
    VkExtent2D                   m_swapchainExtent         {};
    VkSurfaceFormatKHR           m_swapchainSurfaceFormat  {};
    std::vector<VkImage>         m_swapchainImages         {};
    std::vector<VkImageView>     m_swapchainImageViews     {};
    // depth testing
    VkImage                      m_depthImage              {VK_NULL_HANDLE};
    VkImageView                  m_depthImageView          {VK_NULL_HANDLE};
    VmaAllocation                m_depthImageAllocation    {VK_NULL_HANDLE};
    // pipeline & rendering
    VkPipelineLayout             m_pipelineLayout          {VK_NULL_HANDLE};
    VkPipeline                   m_graphicsPipeline        {VK_NULL_HANDLE};
    VkCommandPool                m_commandPool             {VK_NULL_HANDLE};
    std::vector<VkCommandBuffer> m_commandBuffers          {};
    std::vector<VkFence>         m_renderWaitFences        {};
    std::vector<VkSemaphore>     m_acquireImageSemaphores  {};
    std::vector<VkSemaphore>     m_renderFinishedSemaphores{};
    uint32_t                     m_frameIndex              {};
    // model data
    VkBuffer                     m_vertexBuffer            {VK_NULL_HANDLE};
    VmaAllocation                m_vertexAllocation        {VK_NULL_HANDLE};
    VmaAllocationInfo            m_vertexAllocationInfo    {};
    VkDeviceAddress              m_vertexBufferAddress     {};
    VkBuffer                     m_indexBuffer             {VK_NULL_HANDLE};
    uint32_t                     m_indexCount              {};
    VmaAllocation                m_indexAllocation         {VK_NULL_HANDLE};
    VmaAllocationInfo            m_indexAllocationInfo     {};
    // descriptors
    VkDescriptorSetLayout        m_descriptorSetLayout     {VK_NULL_HANDLE};
    VkDescriptorPool             m_descriptorPool          {VK_NULL_HANDLE};
    VkDescriptorSet              m_descriptorSet           {VK_NULL_HANDLE};
    // textures
    uint32_t                     m_nextTextureSlot         {};
    VkSampler                    m_textureSampler          {VK_NULL_HANDLE};
    std::vector<VkImage>         m_textureImages           {};
    std::vector<VmaAllocation>   m_textureAllocations      {};
    std::vector<VkImageView>     m_textureImageViews       {};
    // materials
    VkBuffer                     m_materialBuffer          {VK_NULL_HANDLE};
    VmaAllocation                m_materialAllocation      {VK_NULL_HANDLE};
    VmaAllocationInfo            m_materialAllocationInfo  {};
    VkDeviceAddress              m_materialBufferAddress   {};
    // velocity render target (owned by MotionVectors, set via setVelocityTarget)
    VkImageView                  m_velocityImageView       {VK_NULL_HANDLE};
    VkImage                      m_velocityImage           {VK_NULL_HANDLE};
    // extensions
    std::vector<const char *>    m_requiredDeviceExtensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        "VK_KHR_portability_subset",
    };

    void createInstance(Window &window);
    void pickPhysicalDevice();
    void createDevice();
    void createAllocator();
    void createSurface(Window &window);
    void createSwapchain(Window &window);
    void createImageViews();
    void createDepthResources();
    void createGraphicsPipeline();
    void createCommandPool();
    void createCommandBuffers();
    void createSyncObjects();
    void createBuffer(VkBuffer &buffer, VkDeviceSize size, VkBufferUsageFlags usage, VmaAllocation &allocation, VmaAllocationInfo &allocationInfo);
    void createDescriptorSet();
    void createTextureSampler();
    void recreateSwapchain(Window &window);

    std::vector<char> readFile(const std::string &filename);
    void checkResult(VkResult result, const char *errorMessage) const;
    void swapchainCleanup();
    void syncObjectCleanup();
    void vmaCleanup();
};
