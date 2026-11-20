[[vk::binding(0, 0)]] Sampler2D inputTex;
[[vk::binding(1, 0)]] RWTexture2D<float4> outputTex;

struct PushConstants {
    float sharpenStrength;
};

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    int2 coord = int2(dispatchThreadID.xy);
    uint2 texSize;
    outputTex.GetDimensions(texSize.x, texSize.y);

    if (coord.x >= (int)texSize.x || coord.y >= (int)texSize.y)
        return;

    float2 uv = (float2(coord) + 0.5) / float2(texSize);
    float2 texelSize = 1.0 / float2(texSize);

    // Sample cross pattern
    float3 center = inputTex.SampleLevel(uv, 0).rgb;
    float3 north  = inputTex.SampleLevel(uv + float2(0, -texelSize.y), 0).rgb;
    float3 south  = inputTex.SampleLevel(uv + float2(0,  texelSize.y), 0).rgb;
    float3 west   = inputTex.SampleLevel(uv + float2(-texelSize.x, 0), 0).rgb;
    float3 east   = inputTex.SampleLevel(uv + float2( texelSize.x, 0), 0).rgb;

    // Compute local contrast
    float3 minColor = min(center, min(min(north, south), min(west, east)));
    float3 maxColor = max(center, max(max(north, south), max(west, east)));

    // RCAS weight: higher contrast = less sharpening (adaptive)
    float3 range = maxColor - minColor;
    float3 rcpRange = 1.0 / (range + 0.001);

    // Sharpen: subtract neighbors, amplify center
    float3 sharpened = center + (4.0 * center - north - south - west - east) * pc.sharpenStrength * min(rcpRange, 1.0);

    outputTex[coord] = float4(clamp(sharpened, 0.0, 1.0), 1.0);
}
