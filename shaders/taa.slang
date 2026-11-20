[[vk::binding(0, 0)]] Sampler2D colorTex;
[[vk::binding(1, 0)]] Sampler2D velocityTex;
[[vk::binding(2, 0)]] Sampler2D depthTex;
[[vk::binding(3, 0)]] Sampler2D historyTex;
[[vk::binding(4, 0)]] RWTexture2D<float4> outputTex;
[[vk::binding(5, 0)]] RWTexture2D<float4> historyWriteTex;

struct PushConstants {
    float blendFactor;
};

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    int2 pixelCoord = int2(dispatchThreadID.xy);
    uint2 texSize;
    outputTex.GetDimensions(texSize.x, texSize.y);

    if (pixelCoord.x >= texSize.x || pixelCoord.y >= texSize.y)
        return;

    float2 uv = (float2(pixelCoord) + 0.5) / float2(texSize);

    // Read current frame data
    float4 currentColor = colorTex.SampleLevel(uv, 0);
    float2 velocity = velocityTex.SampleLevel(uv, 0).rg;

    // Reproject: where was this pixel last frame?
    float2 historyUV = uv - velocity;

    // Reject if reprojected UV is out of bounds
    bool validHistory = all(historyUV >= 0.0) && all(historyUV <= 1.0);

    float4 historyColor = historyTex.SampleLevel(historyUV, 0);

    // Neighborhood clamping (3x3 min/max)
    float3 neighborMin = currentColor.rgb;
    float3 neighborMax = currentColor.rgb;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float2 neighborUV = (float2(pixelCoord + int2(dx, dy)) + 0.5) / float2(texSize);
            float3 neighborColor = colorTex.SampleLevel(neighborUV, 0).rgb;
            neighborMin = min(neighborMin, neighborColor);
            neighborMax = max(neighborMax, neighborColor);
        }
    }

    // Clamp history to neighborhood bounds
    historyColor.rgb = clamp(historyColor.rgb, neighborMin, neighborMax);

    // Blend
    float4 resolved;
    if (validHistory) {
        resolved = float4(lerp(historyColor.rgb, currentColor.rgb, pc.blendFactor), 1.0);
    } else {
        resolved = currentColor;
    }

    outputTex[pixelCoord] = resolved;
    historyWriteTex[pixelCoord] = resolved;
}
