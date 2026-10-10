[[vk::binding(0, 0)]] Sampler2D colorTex;      // low-res
[[vk::binding(1, 0)]] Sampler2D velocityTex;    // low-res
[[vk::binding(2, 0)]] Sampler2D depthTex;       // low-res
[[vk::binding(3, 0)]] Sampler2D historyTex;     // full-res
[[vk::binding(4, 0)]] RWTexture2D<float4> outputTex;       // full-res
[[vk::binding(5, 0)]] RWTexture2D<float4> historyWriteTex; // full-res

struct PushConstants {
    float blendFactor;
    float upscaleRatio;
};

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    int2 outputCoord = int2(dispatchThreadID.xy);
    uint2 outputSize;
    outputTex.GetDimensions(outputSize.x, outputSize.y);

    if (outputCoord.x >= (int)outputSize.x || outputCoord.y >= (int)outputSize.y)
        return;

    float2 outputUV = (float2(outputCoord) + 0.5) / float2(outputSize);

    // Sample low-res inputs at the corresponding position
    float4 currentColor = colorTex.SampleLevel(outputUV, 0);
    float2 velocity = velocityTex.SampleLevel(outputUV, 0).rg;
    float currentDepth = depthTex.SampleLevel(outputUV, 0).r;

    // Reproject into history
    float2 historyUV = outputUV - velocity;
    bool validHistory = all(historyUV >= 0.0) && all(historyUV <= 1.0);

    float4 historyColor = historyTex.SampleLevel(historyUV, 0);

    // Disocclusion detection: check depth discontinuity
    // Compare current depth with history depth (stored in alpha of history)
    float historyDepth = historyColor.a;
    float depthDiff = abs(currentDepth - historyDepth);
    bool disoccluded = depthDiff > 0.01 || !validHistory;

    // Neighborhood clamping on low-res grid
    uint2 inputSize = uint2(float2(outputSize) * pc.upscaleRatio);
    int2 inputCoord = int2(float2(outputCoord) * pc.upscaleRatio);

    float3 neighborMin = currentColor.rgb;
    float3 neighborMax = currentColor.rgb;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float2 neighborUV = (float2(inputCoord + int2(dx, dy)) + 0.5) / float2(inputSize);
            float3 nc = colorTex.SampleLevel(neighborUV, 0).rgb;
            neighborMin = min(neighborMin, nc);
            neighborMax = max(neighborMax, nc);
        }
    }
    historyColor.rgb = clamp(historyColor.rgb, neighborMin, neighborMax);

    // Blend — weight current frame more heavily on disocclusion
    float blend = disoccluded ? 1.0 : pc.blendFactor;
    float4 resolved = float4(lerp(historyColor.rgb, currentColor.rgb, blend), currentDepth);

    outputTex[outputCoord] = resolved;
    historyWriteTex[outputCoord] = resolved;
}
