#include "Renderer.h"
#include "PushConstants.h"
#include "Config.h"
#include "Window/Window.h"
#include "Camera/Camera.h"
#include "PostProcess/PostProcessPipeline.h"
#include "MotionVectors/MotionVectors.h"
#include "volk.h"
#include "vk_mem_alloc.h"
#include "stb_image.h"
#include "Vertex.h"
#include "Material.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <ranges>
#include <string>
#include <fstream>
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <vector>
#include <queue>
#include <print>
#include <cstring>
#include <cmath>
#include <array>

// ─── Accessors ───────────────────────────────────────────────────────────────

VkDevice          Renderer::getDevice()          const { return m_device; }
VkPhysicalDevice  Renderer::getPhysicalDevice()  const { return m_physicalDevice; }
VmaAllocator      Renderer::getAllocator()        const { return m_allocator; }
VkQueue           Renderer::getQueue()            const { return m_queue; }
uint32_t          Renderer::getQueueIndex()       const { return m_queueIndex; }
VkCommandPool     Renderer::getCommandPool()      const { return m_commandPool; }
VkExtent2D        Renderer::getSwapchainExtent()  const { return m_swapchainExtent; }
VkFormat          Renderer::getSwapchainFormat()  const { return m_swapchainSurfaceFormat.format; }
VkImageView       Renderer::getColorImageView()   const { return m_colorImageView; }
VkImage           Renderer::getColorImage()       const { return m_colorImage; }
VkImageView       Renderer::getDepthImageView()   const { return m_depthImageView; }
VkImage           Renderer::getDepthImage()       const { return m_depthImage; }

void Renderer::setVelocityTarget(VkImageView velocityImageView, VkImage velocityImage) {
    m_velocityImageView = velocityImageView;
    m_velocityImage     = velocityImage;
}

void Renderer::setInternalResolution(float ratio) {
    m_internalResolutionRatio = ratio;
}

VkExtent2D Renderer::getInternalExtent() const {
    return {
        static_cast<uint32_t>(m_swapchainExtent.width  * m_internalResolutionRatio),
        static_cast<uint32_t>(m_swapchainExtent.height * m_internalResolutionRatio)
    };
}

// ─── Public init ─────────────────────────────────────────────────────────────

void Renderer::initVulkan(Window &window) {
    checkResult(volkInitialize(), "Error: failed to initialize Volk");
    createInstance(window);
    createSurface(window);
    pickPhysicalDevice();
    createDevice();
    createAllocator();
    createSwapchain(window);
    createImageViews();
    createColorResources();
    createDepthResources();
    createDescriptorSet();
    createTextureSampler();
    createGraphicsPipeline();
    createCommandPool();
    createCommandBuffers();
    createSyncObjects();
}

// ─── Upload helpers ───────────────────────────────────────────────────────────

void Renderer::uploadModel(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices) {
    createBuffer(
        m_vertexBuffer,
        vertices.size() * sizeof(Vertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_vertexAllocation,
        m_vertexAllocationInfo
    );
    createBuffer(
        m_indexBuffer,
        indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_indexAllocation,
        m_indexAllocationInfo
    );

    memcpy(m_vertexAllocationInfo.pMappedData, vertices.data(), vertices.size() * sizeof(Vertex));
    memcpy(m_indexAllocationInfo.pMappedData, indices.data(), indices.size() * sizeof(uint32_t));

    VkBufferDeviceAddressInfo bufferDeviceAddressInfo{
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext  = nullptr,
        .buffer = m_vertexBuffer
    };
    m_vertexBufferAddress = vkGetBufferDeviceAddress(m_device, &bufferDeviceAddressInfo);
    m_indexCount = static_cast<uint32_t>(indices.size());
}

uint32_t Renderer::uploadTexture(const std::string &path, VkFormat format) {
    int width{}, height{}, channels{};
    stbi_uc *pixels{stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha)};
    if (pixels == nullptr) {
        throw std::runtime_error(std::format("Error: failed to load texture: {}", path));
    }

    VkDeviceSize imageSize{static_cast<VkDeviceSize>(width * height * 4)};

    // Create staging buffer
    VkBuffer stagingBuffer{VK_NULL_HANDLE};
    VmaAllocation stagingAllocation{VK_NULL_HANDLE};
    VmaAllocationInfo stagingAllocInfo{};
    createBuffer(stagingBuffer, imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        stagingAllocation, stagingAllocInfo);
    memcpy(stagingAllocInfo.pMappedData, pixels, static_cast<size_t>(imageSize));
    stbi_image_free(pixels);

    // Create device-local image
    VkImageCreateInfo imageCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = {},
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = format,
        .extent        = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1},
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    VmaAllocationCreateInfo allocationCreateInfo{
        .usage = VMA_MEMORY_USAGE_AUTO
    };

    VkImage image{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    checkResult(vmaCreateImage(m_allocator, &imageCreateInfo, &allocationCreateInfo, &image, &allocation, nullptr),
        "Error: failed to create texture image");

    // Transition + copy via one-time command buffer
    VkCommandBuffer cmd{};
    VkCommandBufferAllocateInfo cmdAllocInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = m_commandPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    vkAllocateCommandBuffers(m_device, &cmdAllocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vkBeginCommandBuffer(cmd, &beginInfo);

    // UNDEFINED → TRANSFER_DST
    VkImageMemoryBarrier2 toTransferBarrier{
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext               = nullptr,
        .srcStageMask        = VK_PIPELINE_STAGE_2_NONE,
        .srcAccessMask       = VK_ACCESS_2_NONE,
        .dstStageMask        = VK_PIPELINE_STAGE_2_COPY_BIT,
        .dstAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = image,
        .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    VkDependencyInfo depInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext                   = nullptr,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &toTransferBarrier
    };
    vkCmdPipelineBarrier2(cmd, &depInfo);

    // Copy buffer → image
    VkBufferImageCopy region{
        .bufferOffset      = 0,
        .bufferRowLength   = 0,
        .bufferImageHeight = 0,
        .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset       = {0, 0, 0},
        .imageExtent       = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1}
    };
    vkCmdCopyBufferToImage(cmd, stagingBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // TRANSFER_DST → SHADER_READ_ONLY
    VkImageMemoryBarrier2 toReadBarrier{
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext               = nullptr,
        .srcStageMask        = VK_PIPELINE_STAGE_2_COPY_BIT,
        .srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask        = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dstAccessMask       = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        .oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = image,
        .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    VkDependencyInfo depInfo2{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext                   = nullptr,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &toReadBarrier
    };
    vkCmdPipelineBarrier2(cmd, &depInfo2);

    vkEndCommandBuffer(cmd);

    VkCommandBufferSubmitInfo cmdSubmitInfo{
        .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext         = nullptr,
        .commandBuffer = cmd,
        .deviceMask    = 0
    };
    VkSubmitInfo2 submitInfo{
        .sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext                  = nullptr,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos    = &cmdSubmitInfo
    };
    vkQueueSubmit2(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_queue);

    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);
    vmaDestroyBuffer(m_allocator, stagingBuffer, stagingAllocation);

    // Create image view
    VkImageViewCreateInfo imageViewCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = {},
        .image            = image,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = format,
        .components       = {},
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    VkImageView imageView{VK_NULL_HANDLE};
    checkResult(vkCreateImageView(m_device, &imageViewCreateInfo, nullptr, &imageView),
        "Error: failed to create texture image view");

    // Write into bindless array
    uint32_t slot{m_nextTextureSlot++};
    VkDescriptorImageInfo descriptorImageInfo{
        .sampler     = m_textureSampler,
        .imageView   = imageView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    };
    VkWriteDescriptorSet writeDescriptorSet{
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext           = nullptr,
        .dstSet          = m_descriptorSet,
        .dstBinding      = 0,
        .dstArrayElement = slot,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo      = &descriptorImageInfo
    };
    vkUpdateDescriptorSets(m_device, 1, &writeDescriptorSet, 0, nullptr);

    m_textureImages.push_back(image);
    m_textureAllocations.push_back(allocation);
    m_textureImageViews.push_back(imageView);
    return slot;
}

void Renderer::uploadMaterials(std::vector<Material> &materials) {
    createBuffer(
        m_materialBuffer,
        materials.size() * sizeof(Material),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_materialAllocation,
        m_materialAllocationInfo
    );
    memcpy(m_materialAllocationInfo.pMappedData, materials.data(), materials.size() * sizeof(Material));

    VkBufferDeviceAddressInfo bufferDeviceAddressInfo{
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext  = nullptr,
        .buffer = m_materialBuffer
    };
    m_materialBufferAddress = vkGetBufferDeviceAddress(m_device, &bufferDeviceAddressInfo);
}

// ─── Instance creation ────────────────────────────────────────────────────────

void Renderer::createInstance(Window &window) {
    constexpr VkApplicationInfo applicationInfo{
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext              = nullptr,
        .pApplicationName   = Config::title,
        .applicationVersion = Config::version,
        .pEngineName        = nullptr,
        .engineVersion      = {},
        .apiVersion         = VK_API_VERSION_1_3,
    };

    auto [glfwExtensionCount, glfwExtensions] = window.getRequiredInstanceExtensions();

    std::vector<const char *> instanceExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    const VkInstanceCreateInfo instanceCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext                   = nullptr,
        .flags                   = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
        .pApplicationInfo        = &applicationInfo,
        .enabledLayerCount       = 0,
        .ppEnabledLayerNames     = nullptr,
        .enabledExtensionCount   = static_cast<uint32_t>(instanceExtensions.size()),
        .ppEnabledExtensionNames = instanceExtensions.data()
    };

    checkResult(vkCreateInstance(&instanceCreateInfo, nullptr, &m_instance),
        "Error: failed to create Vulkan instance");
    volkLoadInstance(m_instance);
}

// ─── Surface ─────────────────────────────────────────────────────────────────

void Renderer::createSurface(Window &window) {
    window.createWindowSurface(m_instance, nullptr, &m_surface);
}

// ─── Physical device selection ────────────────────────────────────────────────

void Renderer::pickPhysicalDevice() {
    uint32_t physicalDeviceCount{};
    checkResult(vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, nullptr),
        "Error: failed to enumerate physical device count");
    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    checkResult(vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, physicalDevices.data()),
        "Error: failed to fill physicalDevices vector");

    std::priority_queue<std::pair<int, VkPhysicalDevice>> priorityQueue{};
    for (VkPhysicalDevice physicalDevice : physicalDevices) {
        VkPhysicalDeviceProperties deviceProperties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

        int score{};
        switch (deviceProperties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                score += 2;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                score += 1;
                break;
            default:
                continue;
        }
        priorityQueue.push({score, physicalDevice});
    }

    if (!priorityQueue.empty()) {
        m_physicalDevice = priorityQueue.top().second;
    } else {
        throw std::runtime_error("Error: failed to find compatible GPU");
    }

    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(m_physicalDevice, &deviceProperties);
    std::println("Selected GPU: {}", deviceProperties.deviceName);
}

// ─── Logical device ───────────────────────────────────────────────────────────

void Renderer::createDevice() {
    uint32_t queueFamilyPropertyCount{};
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyPropertyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyPropertyCount, queueFamilyProperties.data());

    for (uint32_t i{}; i < static_cast<uint32_t>(queueFamilyProperties.size()); ++i) {
        if (queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            m_queueIndex = i;
            break;
        }
    }

    float queuePriority{1.0f};
    VkDeviceQueueCreateInfo queueCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = {},
        .queueFamilyIndex = m_queueIndex,
        .queueCount       = 1,
        .pQueuePriorities = &queuePriority
    };

    VkPhysicalDeviceVulkan11Features vulkan11Features{
        .sType                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext                = nullptr,
        .shaderDrawParameters = VK_TRUE
    };

    VkPhysicalDeviceVulkan12Features vulkan12Features{
        .sType                                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext                                     = &vulkan11Features,
        .descriptorIndexing                        = VK_TRUE,
        .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
        .descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
        .descriptorBindingPartiallyBound           = VK_TRUE,
        .runtimeDescriptorArray                    = VK_TRUE,
        .scalarBlockLayout                         = VK_TRUE,
        .bufferDeviceAddress                       = VK_TRUE,
    };

    VkPhysicalDeviceVulkan13Features vulkan13Features{
        .sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext            = &vulkan12Features,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE
    };

    VkPhysicalDeviceFeatures deviceFeatures{
        .independentBlend  = VK_TRUE,
        .samplerAnisotropy = VK_TRUE
    };

    VkDeviceCreateInfo deviceCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &vulkan13Features,
        .queueCreateInfoCount    = 1,
        .pQueueCreateInfos       = &queueCreateInfo,
        .enabledExtensionCount   = static_cast<uint32_t>(m_requiredDeviceExtensions.size()),
        .ppEnabledExtensionNames = m_requiredDeviceExtensions.data(),
        .pEnabledFeatures        = &deviceFeatures
    };

    checkResult(vkCreateDevice(m_physicalDevice, &deviceCreateInfo, nullptr, &m_device),
        "Error: failed to create Vulkan logical device");
    vkGetDeviceQueue(m_device, m_queueIndex, 0, &m_queue);
    volkLoadDevice(m_device);
}

// ─── VMA allocator ────────────────────────────────────────────────────────────

void Renderer::createAllocator() {
    VmaAllocatorCreateInfo allocatorCreateInfo{
        .flags                       = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice              = m_physicalDevice,
        .device                      = m_device,
        .preferredLargeHeapBlockSize = {},
        .pAllocationCallbacks        = nullptr,
        .pDeviceMemoryCallbacks      = nullptr,
        .pHeapSizeLimit              = nullptr,
        .pVulkanFunctions            = {},
        .instance                    = m_instance,
        .vulkanApiVersion            = VK_API_VERSION_1_3,
    };

    // Manually populate VmaVulkanFunctions from volk's loaded pointers
    // (VMA_STATIC_VULKAN_FUNCTIONS=0 and VMA_DYNAMIC_VULKAN_FUNCTIONS=0 are set,
    //  so we supply the function pointers explicitly)
    VmaVulkanFunctions vulkanFunctions{
        .vkGetInstanceProcAddr                  = vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr                    = vkGetDeviceProcAddr,
        .vkGetPhysicalDeviceProperties          = vkGetPhysicalDeviceProperties,
        .vkGetPhysicalDeviceMemoryProperties    = vkGetPhysicalDeviceMemoryProperties,
        .vkAllocateMemory                       = vkAllocateMemory,
        .vkFreeMemory                           = vkFreeMemory,
        .vkMapMemory                            = vkMapMemory,
        .vkUnmapMemory                          = vkUnmapMemory,
        .vkFlushMappedMemoryRanges              = vkFlushMappedMemoryRanges,
        .vkInvalidateMappedMemoryRanges         = vkInvalidateMappedMemoryRanges,
        .vkBindBufferMemory                     = vkBindBufferMemory,
        .vkBindImageMemory                      = vkBindImageMemory,
        .vkGetBufferMemoryRequirements          = vkGetBufferMemoryRequirements,
        .vkGetImageMemoryRequirements           = vkGetImageMemoryRequirements,
        .vkCreateBuffer                         = vkCreateBuffer,
        .vkDestroyBuffer                        = vkDestroyBuffer,
        .vkCreateImage                          = vkCreateImage,
        .vkDestroyImage                         = vkDestroyImage,
        .vkCmdCopyBuffer                        = vkCmdCopyBuffer,
        .vkGetBufferMemoryRequirements2KHR      = vkGetBufferMemoryRequirements2,
        .vkGetImageMemoryRequirements2KHR       = vkGetImageMemoryRequirements2,
        .vkBindBufferMemory2KHR                 = vkBindBufferMemory2,
        .vkBindImageMemory2KHR                  = vkBindImageMemory2,
        .vkGetPhysicalDeviceMemoryProperties2KHR= vkGetPhysicalDeviceMemoryProperties2,
        .vkGetDeviceBufferMemoryRequirements    = vkGetDeviceBufferMemoryRequirements,
        .vkGetDeviceImageMemoryRequirements     = vkGetDeviceImageMemoryRequirements,
    };
    allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;

    checkResult(vmaCreateAllocator(&allocatorCreateInfo, &m_allocator),
        "Error: Failed to create VMA Allocator");
}

// ─── Swapchain ────────────────────────────────────────────────────────────────

void Renderer::createSwapchain(Window &window) {
    // Query surface capabilities
    VkSurfaceCapabilitiesKHR surfaceCapabilities{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &surfaceCapabilities);

    // Choose surface format (prefer SRGB B8G8R8A8)
    uint32_t formatCount{};
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &formatCount, surfaceFormats.data());

    m_swapchainSurfaceFormat = surfaceFormats[0];
    for (const auto &surfaceFormat : surfaceFormats) {
        if (surfaceFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
            surfaceFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            m_swapchainSurfaceFormat = surfaceFormat;
            break;
        }
    }

    // Choose present mode (prefer mailbox, fall back to FIFO)
    uint32_t presentModeCount{};
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount, presentModes.data());

    // FIFO, not mailbox: frame interpolation queues two presents back to back, and mailbox
    // would replace the interpolated frame with the real one before it is ever displayed
    VkPresentModeKHR presentMode{VK_PRESENT_MODE_FIFO_KHR};

    // Choose extent
    if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        m_swapchainExtent = surfaceCapabilities.currentExtent;
    } else {
        auto [fbWidth, fbHeight] = window.getFramebufferSize();
        m_swapchainExtent = {
            std::clamp(static_cast<uint32_t>(fbWidth),
                surfaceCapabilities.minImageExtent.width,
                surfaceCapabilities.maxImageExtent.width),
            std::clamp(static_cast<uint32_t>(fbHeight),
                surfaceCapabilities.minImageExtent.height,
                surfaceCapabilities.maxImageExtent.height)
        };
    }

    uint32_t imageCount{surfaceCapabilities.minImageCount + 1};
    if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount) {
        imageCount = surfaceCapabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swapchainCreateInfo{
        .sType                 = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext                 = nullptr,
        .flags                 = {},
        .surface               = m_surface,
        .minImageCount         = imageCount,
        .imageFormat           = m_swapchainSurfaceFormat.format,
        .imageColorSpace       = m_swapchainSurfaceFormat.colorSpace,
        .imageExtent           = m_swapchainExtent,
        .imageArrayLayers      = 1,
        .imageUsage            = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .preTransform          = surfaceCapabilities.currentTransform,
        .compositeAlpha        = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode           = presentMode,
        .clipped               = VK_TRUE,
        .oldSwapchain          = VK_NULL_HANDLE
    };

    checkResult(vkCreateSwapchainKHR(m_device, &swapchainCreateInfo, nullptr, &m_swapchain),
        "Error: failed to create swapchain");

    uint32_t swapchainImageCount{};
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &swapchainImageCount, nullptr);
    m_swapchainImages.resize(swapchainImageCount);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &swapchainImageCount, m_swapchainImages.data());
}

// ─── Image views ─────────────────────────────────────────────────────────────

void Renderer::createImageViews() {
    m_swapchainImageViews.resize(m_swapchainImages.size());
    for (size_t i{}; i < m_swapchainImages.size(); ++i) {
        VkImageViewCreateInfo imageViewCreateInfo{
            .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext            = nullptr,
            .flags            = {},
            .image            = m_swapchainImages[i],
            .viewType         = VK_IMAGE_VIEW_TYPE_2D,
            .format           = m_swapchainSurfaceFormat.format,
            .components       = {},
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        };
        checkResult(vkCreateImageView(m_device, &imageViewCreateInfo, nullptr, &m_swapchainImageViews[i]),
            "Error: failed to create swapchain image view");
    }
}

// ─── Offscreen color resources ───────────────────────────────────────────────

void Renderer::createColorResources() {
    VkExtent2D renderExtent{getInternalExtent()};

    VkImageCreateInfo imageCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = {},
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = VK_FORMAT_R16G16B16A16_SFLOAT,
        .extent        = {renderExtent.width, renderExtent.height, 1},
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    VmaAllocationCreateInfo allocationCreateInfo{
        .usage = VMA_MEMORY_USAGE_AUTO
    };

    checkResult(vmaCreateImage(m_allocator, &imageCreateInfo, &allocationCreateInfo,
        &m_colorImage, &m_colorImageAllocation, nullptr),
        "Error: failed to create offscreen color image");

    VkImageViewCreateInfo imageViewCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = {},
        .image            = m_colorImage,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = VK_FORMAT_R16G16B16A16_SFLOAT,
        .components       = {},
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };

    checkResult(vkCreateImageView(m_device, &imageViewCreateInfo, nullptr, &m_colorImageView),
        "Error: failed to create offscreen color image view");
}

// ─── Depth resources ─────────────────────────────────────────────────────────

void Renderer::createDepthResources() {
    VkFormat depthFormat{VK_FORMAT_D32_SFLOAT};
    VkExtent2D renderExtent{getInternalExtent()};

    VkImageCreateInfo imageCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = {},
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = depthFormat,
        .extent        = {renderExtent.width, renderExtent.height, 1},
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    VmaAllocationCreateInfo allocationCreateInfo{
        .usage = VMA_MEMORY_USAGE_AUTO
    };

    checkResult(vmaCreateImage(m_allocator, &imageCreateInfo, &allocationCreateInfo,
        &m_depthImage, &m_depthImageAllocation, nullptr),
        "Error: failed to create depth image");

    VkImageViewCreateInfo imageViewCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = {},
        .image            = m_depthImage,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = depthFormat,
        .components       = {},
        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}
    };

    checkResult(vkCreateImageView(m_device, &imageViewCreateInfo, nullptr, &m_depthImageView),
        "Error: failed to create depth image view");
}

// ─── Descriptor set ───────────────────────────────────────────────────────────

void Renderer::createDescriptorSet() {
    // Layout: binding 0 = bindless combined image sampler array
    VkDescriptorSetLayoutBinding binding{
        .binding            = 0,
        .descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount    = static_cast<uint32_t>(Config::maxBindlessTextures),
        .stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = nullptr
    };

    VkDescriptorBindingFlags bindingFlags{
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
    };

    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .pNext         = nullptr,
        .bindingCount  = 1,
        .pBindingFlags = &bindingFlags
    };

    VkDescriptorSetLayoutCreateInfo layoutCreateInfo{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext        = &bindingFlagsCreateInfo,
        .flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = 1,
        .pBindings    = &binding
    };

    checkResult(vkCreateDescriptorSetLayout(m_device, &layoutCreateInfo, nullptr, &m_descriptorSetLayout),
        "Error: failed to create descriptor set layout");

    // Pool
    VkDescriptorPoolSize poolSize{
        .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = static_cast<uint32_t>(Config::maxBindlessTextures)
    };

    VkDescriptorPoolCreateInfo poolCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets       = 1,
        .poolSizeCount = 1,
        .pPoolSizes    = &poolSize
    };

    checkResult(vkCreateDescriptorPool(m_device, &poolCreateInfo, nullptr, &m_descriptorPool),
        "Error: failed to create descriptor pool");

    // Allocate
    VkDescriptorSetAllocateInfo allocateInfo{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext              = nullptr,
        .descriptorPool     = m_descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &m_descriptorSetLayout
    };

    checkResult(vkAllocateDescriptorSets(m_device, &allocateInfo, &m_descriptorSet),
        "Error: failed to allocate descriptor set");
}

// ─── Texture sampler ─────────────────────────────────────────────────────────

void Renderer::createTextureSampler() {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(m_physicalDevice, &properties);

    VkSamplerCreateInfo samplerCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext                   = nullptr,
        .flags                   = {},
        .magFilter               = VK_FILTER_LINEAR,
        .minFilter               = VK_FILTER_LINEAR,
        .mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .mipLodBias              = 0.0f,
        .anisotropyEnable        = VK_TRUE,
        .maxAnisotropy           = properties.limits.maxSamplerAnisotropy,
        .compareEnable           = VK_FALSE,
        .compareOp               = VK_COMPARE_OP_ALWAYS,
        .minLod                  = 0.0f,
        .maxLod                  = VK_LOD_CLAMP_NONE,
        .borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE
    };

    checkResult(vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_textureSampler),
        "Error: failed to create texture sampler");
}

// ─── Graphics pipeline ────────────────────────────────────────────────────────

void Renderer::createGraphicsPipeline() {
    // Load pre-compiled SPIR-V
    std::vector<char> vertCode{readFile(std::string(SHADER_DIR) + "forward_vertex.spv")};
    std::vector<char> fragCode{readFile(std::string(SHADER_DIR) + "forward_fragment.spv")};

    auto createShaderModule = [this](const std::vector<char> &code) {
        VkShaderModuleCreateInfo createInfo{
            .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext    = nullptr,
            .codeSize = code.size(),
            .pCode    = reinterpret_cast<const uint32_t *>(code.data())
        };
        VkShaderModule module{VK_NULL_HANDLE};
        checkResult(vkCreateShaderModule(m_device, &createInfo, nullptr, &module),
            "Error: failed to create shader module");
        return module;
    };

    VkShaderModule vertModule{createShaderModule(vertCode)};
    VkShaderModule fragModule{createShaderModule(fragCode)};

    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{{
        {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext  = nullptr,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertModule,
            .pName  = "main"
        },
        {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext  = nullptr,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragModule,
            .pName  = "main"
        }
    }};

    // No vertex input — we read from buffer device address in vertex shader
    VkPipelineVertexInputStateCreateInfo vertexInputState{
        .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext                           = nullptr,
        .vertexBindingDescriptionCount   = 0,
        .pVertexBindingDescriptions      = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions    = nullptr
    };

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext                  = nullptr,
        .topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE
    };

    VkPipelineViewportStateCreateInfo viewportState{
        .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext         = nullptr,
        .viewportCount = 1,
        .pViewports    = nullptr,
        .scissorCount  = 1,
        .pScissors     = nullptr
    };

    VkPipelineRasterizationStateCreateInfo rasterizationState{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext                   = nullptr,
        .depthClampEnable        = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode             = VK_POLYGON_MODE_FILL,
        .cullMode                = VK_CULL_MODE_BACK_BIT,
        .frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable         = VK_FALSE,
        .lineWidth               = 1.0f
    };

    VkPipelineMultisampleStateCreateInfo multisampleState{
        .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext                = nullptr,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable  = VK_FALSE
    };

    VkPipelineDepthStencilStateCreateInfo depthStencilState{
        .sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext                 = nullptr,
        .depthTestEnable       = VK_TRUE,
        .depthWriteEnable      = VK_TRUE,
        .depthCompareOp        = VK_COMPARE_OP_LESS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable     = VK_FALSE
    };

    // Two blend attachments: [0] = swapchain color, [1] = velocity (RG16F)
    std::array<VkPipelineColorBlendAttachmentState, 2> colorBlendAttachments{{
        {
            .blendEnable    = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
        },
        {
            .blendEnable    = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
        }
    }};

    VkPipelineColorBlendStateCreateInfo colorBlendState{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext           = nullptr,
        .logicOpEnable   = VK_FALSE,
        .attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size()),
        .pAttachments    = colorBlendAttachments.data()
    };

    // Push constants
    VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset     = 0,
        .size       = sizeof(PushConstants)
    };

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext                  = nullptr,
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_descriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstantRange
    };

    checkResult(vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_pipelineLayout),
        "Error: failed to create pipeline layout");

    // Dynamic rendering (no render pass object)
    std::array<VkFormat, 2> colorAttachmentFormats{
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R16G16_SFLOAT
    };

    VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext                   = nullptr,
        .colorAttachmentCount    = static_cast<uint32_t>(colorAttachmentFormats.size()),
        .pColorAttachmentFormats = colorAttachmentFormats.data(),
        .depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT
    };

    std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicStateInfo{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext             = nullptr,
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates    = dynamicStates.data()
    };

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext               = &pipelineRenderingCreateInfo,
        .flags               = {},
        .stageCount          = static_cast<uint32_t>(shaderStages.size()),
        .pStages             = shaderStages.data(),
        .pVertexInputState   = &vertexInputState,
        .pInputAssemblyState = &inputAssemblyState,
        .pViewportState      = &viewportState,
        .pRasterizationState = &rasterizationState,
        .pMultisampleState   = &multisampleState,
        .pDepthStencilState  = &depthStencilState,
        .pColorBlendState    = &colorBlendState,
        .pDynamicState       = &dynamicStateInfo,
        .layout              = m_pipelineLayout,
        .renderPass          = VK_NULL_HANDLE,
        .subpass             = 0
    };

    checkResult(vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &m_graphicsPipeline),
        "Error: failed to create graphics pipeline");

    vkDestroyShaderModule(m_device, vertModule, nullptr);
    vkDestroyShaderModule(m_device, fragModule, nullptr);
}

// ─── Command pool & buffers ───────────────────────────────────────────────────

void Renderer::createCommandPool() {
    VkCommandPoolCreateInfo commandPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = m_queueIndex
    };

    checkResult(vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &m_commandPool),
        "Error: failed to create command pool");
}

void Renderer::createCommandBuffers() {
    m_commandBuffers.resize(Config::maxFramesInFlight);

    VkCommandBufferAllocateInfo allocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = m_commandPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = static_cast<uint32_t>(m_commandBuffers.size())
    };

    checkResult(vkAllocateCommandBuffers(m_device, &allocateInfo, m_commandBuffers.data()),
        "Error: failed to allocate command buffers");
}

// ─── Sync objects ─────────────────────────────────────────────────────────────

void Renderer::createSyncObjects() {
    m_renderWaitFences.resize(Config::maxFramesInFlight);
    m_acquireImageSemaphores.resize(Config::maxFramesInFlight * m_maxPresentsPerFrame);
    // one per swapchain image: presentation may still hold a semaphore after its frame's fence signals
    m_renderFinishedSemaphores.resize(m_swapchainImages.size());

    VkFenceCreateInfo fenceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };

    VkSemaphoreCreateInfo semaphoreCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = {}
    };

    for (size_t i{}; i < Config::maxFramesInFlight; ++i) {
        checkResult(vkCreateFence(m_device, &fenceCreateInfo, nullptr, &m_renderWaitFences[i]),
            "Error: failed to create render fence");
    }
    for (VkSemaphore &semaphore : m_acquireImageSemaphores) {
        checkResult(vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &semaphore),
            "Error: failed to create acquire semaphore");
    }
    for (VkSemaphore &semaphore : m_renderFinishedSemaphores) {
        checkResult(vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &semaphore),
            "Error: failed to create render finished semaphore");
    }
}

// ─── Buffer helper ────────────────────────────────────────────────────────────

void Renderer::createBuffer(VkBuffer &buffer, VkDeviceSize size, VkBufferUsageFlags usage,
    VmaAllocation &allocation, VmaAllocationInfo &allocationInfo)
{
    VkBufferCreateInfo bufferCreateInfo{
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext       = nullptr,
        .size        = size,
        .usage       = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    VmaAllocationCreateInfo allocationCreateInfo{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO
    };

    checkResult(vmaCreateBuffer(m_allocator, &bufferCreateInfo, &allocationCreateInfo,
        &buffer, &allocation, &allocationInfo),
        "Error: failed to create buffer");
}

// ─── drawFrame ────────────────────────────────────────────────────────────────

void Renderer::drawFrame(Window &window, Camera &camera, PostProcessPipeline *postProcess, MotionVectors *motionVectors,
                         PostProcessMode mode) {
    // Wait for this frame slot's fence
    vkWaitForFences(m_device, 1, &m_renderWaitFences[m_frameIndex], VK_TRUE, UINT64_MAX);

    bool usePostProcess{postProcess != nullptr && motionVectors != nullptr && mode != PostProcessMode::Native};

    // Acquire swapchain images — two when presenting an interpolated frame before the real one
    uint32_t presentCount{(usePostProcess && postProcess->willInterpolate(mode)) ? 2u : 1u};
    std::array<uint32_t, m_maxPresentsPerFrame> imageIndices{};
    for (uint32_t i{}; i < presentCount; ++i) {
        VkResult acquireResult{vkAcquireNextImageKHR(
            m_device, m_swapchain, UINT64_MAX,
            m_acquireImageSemaphores[m_frameIndex * m_maxPresentsPerFrame + i],
            VK_NULL_HANDLE, &imageIndices[i])};

        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
            if (i == 0) {
                recreateSwapchain(window);
                return;
            }
            // Keep the image already acquired and present only the real frame this time
            presentCount = 1;
            break;
        }
        if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("Error: failed to acquire swapchain image");
        }
    }

    vkResetFences(m_device, 1, &m_renderWaitFences[m_frameIndex]);

    VkCommandBuffer cmd{m_commandBuffers[m_frameIndex]};
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = {}
    };
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Transition offscreen color + velocity image: UNDEFINED → COLOR_ATTACHMENT_OPTIMAL
    VkImageMemoryBarrier2 colorToAttach{
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext               = nullptr,
        // Last frame's compute passes and blit read these; finish before discarding
        .srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
        .srcAccessMask       = VK_ACCESS_2_NONE,
        .dstStageMask        = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask       = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = m_colorImage,
        .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };

    VkImageMemoryBarrier2 velocityToAttach{
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext               = nullptr,
        // Last frame's compute passes and blit read these; finish before discarding
        .srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
        .srcAccessMask       = VK_ACCESS_2_NONE,
        .dstStageMask        = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask       = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = m_velocityImage,
        .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };

    // Batch offscreen color + velocity barriers together
    std::array<VkImageMemoryBarrier2, 2> preRenderBarriers{colorToAttach, velocityToAttach};
    VkDependencyInfo colorBarrierDep{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext                   = nullptr,
        .imageMemoryBarrierCount = static_cast<uint32_t>(preRenderBarriers.size()),
        .pImageMemoryBarriers    = preRenderBarriers.data()
    };
    vkCmdPipelineBarrier2(cmd, &colorBarrierDep);

    // Depth image: UNDEFINED → DEPTH_ATTACHMENT_OPTIMAL
    VkImageMemoryBarrier2 depthToAttach{
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext               = nullptr,
        // Last frame's compute passes and blit read these; finish before discarding
        .srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
        .srcAccessMask       = VK_ACCESS_2_NONE,
        .dstStageMask        = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
        .dstAccessMask       = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout           = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = m_depthImage,
        .subresourceRange    = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}
    };
    VkDependencyInfo depthBarrierDep{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext                   = nullptr,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &depthToAttach
    };
    vkCmdPipelineBarrier2(cmd, &depthBarrierDep);

    // Begin dynamic rendering — render to offscreen color target
    VkClearValue clearColor{.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};
    VkClearValue clearVelocity{.color = {.float32 = {0.0f, 0.0f, 0.0f, 0.0f}}};
    VkClearValue clearDepth{.depthStencil = {1.0f, 0}};

    VkRenderingAttachmentInfo colorAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext       = nullptr,
        .imageView   = m_colorImageView,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = clearColor
    };

    VkRenderingAttachmentInfo velocityAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext       = nullptr,
        .imageView   = m_velocityImageView,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = clearVelocity
    };

    VkRenderingAttachmentInfo depthAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext       = nullptr,
        .imageView   = m_depthImageView,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = clearDepth
    };

    std::array<VkRenderingAttachmentInfo, 2> colorAttachments{colorAttachment, velocityAttachment};

    VkExtent2D renderExtent{getInternalExtent()};

    VkRenderingInfo renderingInfo{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext                = nullptr,
        .renderArea           = {.offset = {0, 0}, .extent = renderExtent},
        .layerCount           = 1,
        .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
        .pColorAttachments    = colorAttachments.data(),
        .pDepthAttachment     = &depthAttachment
    };

    vkCmdBeginRendering(cmd, &renderingInfo);

    // Only draw if a model has been uploaded
    if (m_indexCount > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

        VkViewport viewport{
            .x        = 0.0f,
            .y        = 0.0f,
            .width    = static_cast<float>(renderExtent.width),
            .height   = static_cast<float>(renderExtent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f
        };
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{
            .offset = {0, 0},
            .extent = renderExtent
        };
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

        vkCmdBindIndexBuffer(cmd, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // Model is ~2 units across, centered at the origin; the camera starts 3 units back
        glm::mat4 model{1.0f};

        PushConstants pushConstants{
            .materialBufferAddress = m_materialBufferAddress,
            .vertexBufferAddress   = m_vertexBufferAddress,
            .model                 = model,
            .view                  = camera.getViewMatrix(),
            .projection            = camera.getProjectionMatrix(),
            .prevViewProjection    = camera.getPreviousVP(),
            .jitterOffset          = camera.getJitterOffset()
        };

        vkCmdPushConstants(cmd, m_pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(PushConstants), &pushConstants);

        vkCmdDrawIndexed(cmd, m_indexCount, 1, 0, 0, 0);
    }

    vkCmdEndRendering(cmd);

    // ─── Post-process: transition forward pass outputs to shader read ────────
    // Color: COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    // Velocity: COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    // Depth: DEPTH_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    std::array<VkImageMemoryBarrier2, 3> postForwardBarriers{{
        {
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask       = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask       = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            .oldLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = m_colorImage,
            .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        },
        {
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask       = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask       = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            .oldLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = m_velocityImage,
            .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        },
        {
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .srcAccessMask       = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask       = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            .oldLayout           = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = m_depthImage,
            .subresourceRange    = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}
        }
    }};
    VkDependencyInfo postForwardDep{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = static_cast<uint32_t>(postForwardBarriers.size()),
        .pImageMemoryBarriers    = postForwardBarriers.data()
    };
    vkCmdPipelineBarrier2(cmd, &postForwardDep);

    // ─── Execute post-process ─────────────────────────────────────────────
    VkImage blitSourceImage{};
    VkImageLayout blitSourceLayout{};
    if (usePostProcess) {
        postProcess->execute(cmd, *this, mode,
                             m_colorImageView, m_depthImageView, motionVectors->getImageView());
        blitSourceImage = postProcess->getOutputImage();
        blitSourceLayout = VK_IMAGE_LAYOUT_GENERAL;
    } else {
        // Native mode or no post-process: blit offscreen color directly
        blitSourceImage = m_colorImage;
        blitSourceLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // ─── Blit presented frames → swapchain ───────────────────────────────────
    // With frame interpolation the generated midpoint frame goes first, then the real frame
    struct PresentSource {
        VkImage               image;
        VkImageLayout         layout;
        VkPipelineStageFlags2 stage;
        VkAccessFlags2        access;
        VkExtent2D            extent;
    };
    PresentSource realFrame{
        .image  = blitSourceImage,
        .layout = blitSourceLayout,
        .stage  = (blitSourceLayout == VK_IMAGE_LAYOUT_GENERAL)
                    ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                    : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .access = (blitSourceLayout == VK_IMAGE_LAYOUT_GENERAL)
                    ? VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                    : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        // Native mode blits the internal-resolution color target straight to the swapchain
        .extent = (blitSourceLayout == VK_IMAGE_LAYOUT_GENERAL) ? m_swapchainExtent : getInternalExtent()
    };
    std::array<PresentSource, m_maxPresentsPerFrame> presentSources{realFrame};
    if (presentCount == 2) {
        presentSources = {
            PresentSource{
                .image  = postProcess->getInterpolatedImage(),
                .layout = VK_IMAGE_LAYOUT_GENERAL,
                .stage  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .extent = m_swapchainExtent
            },
            realFrame
        };
    }

    // Sources → TRANSFER_SRC_OPTIMAL, swapchain images → TRANSFER_DST_OPTIMAL.
    // Swapchain barriers start at BLIT to chain with the acquire semaphore wait.
    std::vector<VkImageMemoryBarrier2> preBlitBarriers{};
    for (uint32_t i{}; i < presentCount; ++i) {
        preBlitBarriers.push_back({
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = presentSources[i].stage,
            .srcAccessMask       = presentSources[i].access,
            .dstStageMask        = VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
            .dstAccessMask       = VK_ACCESS_2_TRANSFER_READ_BIT,
            .oldLayout           = presentSources[i].layout,
            .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = presentSources[i].image,
            .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        });
        preBlitBarriers.push_back({
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = VK_PIPELINE_STAGE_2_BLIT_BIT,
            .srcAccessMask       = VK_ACCESS_2_NONE,
            .dstStageMask        = VK_PIPELINE_STAGE_2_BLIT_BIT,
            .dstAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = m_swapchainImages[imageIndices[i]],
            .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        });
    }
    VkDependencyInfo preBlitDep{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = static_cast<uint32_t>(preBlitBarriers.size()),
        .pImageMemoryBarriers    = preBlitBarriers.data()
    };
    vkCmdPipelineBarrier2(cmd, &preBlitDep);

    // Blit
    for (uint32_t i{}; i < presentCount; ++i) {
        VkExtent2D srcExtent{presentSources[i].extent};
        VkImageBlit2 blitRegion{
            .sType          = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets     = {{0, 0, 0}, {static_cast<int32_t>(srcExtent.width), static_cast<int32_t>(srcExtent.height), 1}},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffsets     = {{0, 0, 0}, {static_cast<int32_t>(m_swapchainExtent.width), static_cast<int32_t>(m_swapchainExtent.height), 1}}
        };
        VkBlitImageInfo2 blitInfo{
            .sType          = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
            .srcImage       = presentSources[i].image,
            .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .dstImage       = m_swapchainImages[imageIndices[i]],
            .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .regionCount    = 1,
            .pRegions       = &blitRegion,
            .filter         = VK_FILTER_LINEAR
        };
        vkCmdBlitImage2(cmd, &blitInfo);
    }

    // The real frame becomes frame N for the next interpolation
    if (usePostProcess && mode == PostProcessMode::FrameInterp) {
        postProcess->recordPreviousFrameCopy(cmd, *this);
    }

    // Transition swapchain images: TRANSFER_DST_OPTIMAL → PRESENT_SRC_KHR
    std::vector<VkImageMemoryBarrier2> presentBarriers{};
    for (uint32_t i{}; i < presentCount; ++i) {
        presentBarriers.push_back({
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext               = nullptr,
            .srcStageMask        = VK_PIPELINE_STAGE_2_BLIT_BIT,
            .srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask        = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
            .dstAccessMask       = VK_ACCESS_2_NONE,
            .oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = m_swapchainImages[imageIndices[i]],
            .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        });
    }
    VkDependencyInfo presentBarrierDep{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext                   = nullptr,
        .imageMemoryBarrierCount = static_cast<uint32_t>(presentBarriers.size()),
        .pImageMemoryBarriers    = presentBarriers.data()
    };
    vkCmdPipelineBarrier2(cmd, &presentBarrierDep);

    vkEndCommandBuffer(cmd);

    // Submit — wait on every acquire, signal one render-finished semaphore per presented image
    std::array<VkSemaphoreSubmitInfo, m_maxPresentsPerFrame> waitSemaphoreInfos{};
    std::array<VkSemaphoreSubmitInfo, m_maxPresentsPerFrame> signalSemaphoreInfos{};
    for (uint32_t i{}; i < presentCount; ++i) {
        waitSemaphoreInfos[i] = {
            .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext     = nullptr,
            .semaphore = m_acquireImageSemaphores[m_frameIndex * m_maxPresentsPerFrame + i],
            .stageMask = VK_PIPELINE_STAGE_2_BLIT_BIT
        };
        signalSemaphoreInfos[i] = {
            .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext     = nullptr,
            .semaphore = m_renderFinishedSemaphores[imageIndices[i]],
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
        };
    }
    VkCommandBufferSubmitInfo cmdSubmitInfo{
        .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext         = nullptr,
        .commandBuffer = cmd,
        .deviceMask    = 0
    };
    VkSubmitInfo2 submitInfo{
        .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext                    = nullptr,
        .waitSemaphoreInfoCount   = presentCount,
        .pWaitSemaphoreInfos      = waitSemaphoreInfos.data(),
        .commandBufferInfoCount   = 1,
        .pCommandBufferInfos      = &cmdSubmitInfo,
        .signalSemaphoreInfoCount = presentCount,
        .pSignalSemaphoreInfos    = signalSemaphoreInfos.data()
    };
    checkResult(vkQueueSubmit2(m_queue, 1, &submitInfo, m_renderWaitFences[m_frameIndex]),
        "Error: failed to submit command buffer");

    // Present in order; under FIFO each is shown for its own vblank
    bool swapchainStale{false};
    for (uint32_t i{}; i < presentCount; ++i) {
        VkPresentInfoKHR presentInfo{
            .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext              = nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &m_renderFinishedSemaphores[imageIndices[i]],
            .swapchainCount     = 1,
            .pSwapchains        = &m_swapchain,
            .pImageIndices      = &imageIndices[i],
            .pResults           = nullptr
        };
        VkResult presentResult{vkQueuePresentKHR(m_queue, &presentInfo)};

        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
            swapchainStale = true;
        } else if (presentResult != VK_SUCCESS) {
            throw std::runtime_error("Error: failed to present swapchain image");
        }
    }
    if (swapchainStale) {
        recreateSwapchain(window);
    }

    m_frameIndex = (m_frameIndex + 1) % Config::maxFramesInFlight;
}

// ─── Swapchain recreation ─────────────────────────────────────────────────────

void Renderer::recreateSwapchain(Window &window) {
    // Handle minimization
    FramebufferSize size{window.getFramebufferSize()};
    while (size.width == 0 || size.height == 0) {
        size = window.getFramebufferSize();
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(m_device);
    swapchainCleanup();
    createSwapchain(window);
    createImageViews();
    createColorResources();
    createDepthResources();
}

// ─── Cleanup ─────────────────────────────────────────────────────────────────

void Renderer::swapchainCleanup() {
    vkDestroyImageView(m_device, m_colorImageView, nullptr);
    vmaDestroyImage(m_allocator, m_colorImage, m_colorImageAllocation);
    m_colorImageView       = VK_NULL_HANDLE;
    m_colorImage           = VK_NULL_HANDLE;
    m_colorImageAllocation = VK_NULL_HANDLE;

    vkDestroyImageView(m_device, m_depthImageView, nullptr);
    vmaDestroyImage(m_allocator, m_depthImage, m_depthImageAllocation);
    m_depthImageView       = VK_NULL_HANDLE;
    m_depthImage           = VK_NULL_HANDLE;
    m_depthImageAllocation = VK_NULL_HANDLE;

    for (VkImageView view : m_swapchainImageViews) {
        vkDestroyImageView(m_device, view, nullptr);
    }
    m_swapchainImageViews.clear();

    vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
    m_swapchain = VK_NULL_HANDLE;
}

void Renderer::syncObjectCleanup() {
    for (size_t i{}; i < Config::maxFramesInFlight; ++i) {
        vkDestroyFence(m_device, m_renderWaitFences[i], nullptr);
    }
    for (VkSemaphore semaphore : m_acquireImageSemaphores) {
        vkDestroySemaphore(m_device, semaphore, nullptr);
    }
    for (VkSemaphore semaphore : m_renderFinishedSemaphores) {
        vkDestroySemaphore(m_device, semaphore, nullptr);
    }
    m_renderWaitFences.clear();
    m_acquireImageSemaphores.clear();
    m_renderFinishedSemaphores.clear();
}

void Renderer::vmaCleanup() {
    // Textures
    for (size_t i{}; i < m_textureImages.size(); ++i) {
        vkDestroyImageView(m_device, m_textureImageViews[i], nullptr);
        vmaDestroyImage(m_allocator, m_textureImages[i], m_textureAllocations[i]);
    }
    m_textureImages.clear();
    m_textureImageViews.clear();
    m_textureAllocations.clear();

    // Buffers
    if (m_vertexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator, m_vertexBuffer, m_vertexAllocation);
        m_vertexBuffer = VK_NULL_HANDLE;
    }
    if (m_indexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator, m_indexBuffer, m_indexAllocation);
        m_indexBuffer = VK_NULL_HANDLE;
    }
    if (m_materialBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator, m_materialBuffer, m_materialAllocation);
        m_materialBuffer = VK_NULL_HANDLE;
    }

    vmaDestroyAllocator(m_allocator);
    m_allocator = VK_NULL_HANDLE;
}

void Renderer::cleanup() {
    vkDeviceWaitIdle(m_device);

    syncObjectCleanup();
    swapchainCleanup();

    vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroySampler(m_device, m_textureSampler, nullptr);
    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);

    vmaCleanup();

    vkDestroyDevice(m_device, nullptr);
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    vkDestroyInstance(m_instance, nullptr);
}

// ─── Utilities ────────────────────────────────────────────────────────────────

std::vector<char> Renderer::readFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(std::format("Error: failed to open file: {}", filename));
    }
    size_t fileSize{static_cast<size_t>(file.tellg())};
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    return buffer;
}

void Renderer::checkResult(VkResult result, const char *errorMessage) const {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(errorMessage);
    }
}
