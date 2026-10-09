#pragma once

namespace Config {
    // Window
    inline constexpr int         width              {1920};
    inline constexpr int         height             {1080};
    inline constexpr const char *title              {"Temporal Upscaling Renderer"};

    // Vulkan
    inline constexpr int         version            {1};
    inline constexpr int         maxFramesInFlight  {2};
    inline constexpr int         maxBindlessTextures{256};

    // Temporal pipeline
    inline constexpr float       upscaleRatio       {0.667f};
    inline constexpr int         jitterSequenceLen  {16};
    inline constexpr float       taaBlendFactor     {0.05f};
    inline constexpr float       sharpenStrength    {0.5f};
}
