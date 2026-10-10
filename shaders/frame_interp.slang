[[vk::binding(0, 0)]] Sampler2D frameNTex;            // previous resolved frame (full-res)
[[vk::binding(1, 0)]] Sampler2D frameN1Tex;           // current resolved frame (full-res)
[[vk::binding(2, 0)]] Sampler2D velocityTex;          // current - previous, in UV units (low-res)
[[vk::binding(3, 0)]] Sampler2D depthTex;             // low-res
[[vk::binding(4, 0)]] RWTexture2D<float4> outputTex;  // interpolated midpoint frame (full-res)

// Above this RGB distance the two warped samples are treated as disagreeing (disocclusion)
static const float g_disagreementThreshold = 0.25;

// Velocity of the closest surface in a 3x3 neighborhood, so silhouettes move with the
// foreground object instead of smearing the background across them
float2 dilatedVelocity(float2 uv) {
    uint2 depthSize;
    depthTex.GetDimensions(depthSize.x, depthSize.y);
    float2 texel = 1.0 / float2(depthSize);

    float closestDepth = 1.0;
    float2 closestUV = uv;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            float2 sampleUV = uv + float2(x, y) * texel;
            float depth = depthTex.SampleLevel(sampleUV, 0).r;
            if (depth < closestDepth) {
                closestDepth = depth;
                closestUV = sampleUV;
            }
        }
    }
    return velocityTex.SampleLevel(closestUV, 0).rg;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    int2 coord = int2(dispatchThreadID.xy);
    uint2 texSize;
    outputTex.GetDimensions(texSize.x, texSize.y);

    if (coord.x >= texSize.x || coord.y >= texSize.y)
        return;

    float2 uv = (float2(coord) + 0.5) / float2(texSize);
    float2 velocity = dilatedVelocity(uv);

    // Content at the midpoint was half a step back in frame N and half a step ahead in frame N+1
    float2 uvInN  = uv - velocity * 0.5;
    float2 uvInN1 = uv + velocity * 0.5;

    bool validN  = all(uvInN  >= 0.0) && all(uvInN  <= 1.0);
    bool validN1 = all(uvInN1 >= 0.0) && all(uvInN1 <= 1.0);

    float4 sampleN  = frameNTex.SampleLevel(uvInN, 0);
    float4 sampleN1 = frameN1Tex.SampleLevel(uvInN1, 0);

    float4 result;
    if (validN && validN1) {
        // Disagreement means one side sampled a surface that was occluded in that frame;
        // trust the newer frame rather than ghosting the two together
        bool disoccluded = length(sampleN.rgb - sampleN1.rgb) > g_disagreementThreshold;
        result = disoccluded ? sampleN1 : (sampleN + sampleN1) * 0.5;
    } else if (validN1) {
        result = sampleN1;
    } else if (validN) {
        result = sampleN;
    } else {
        // Both warps left the screen: no motion information, fall back to the current frame
        result = frameN1Tex.SampleLevel(uv, 0);
    }

    outputTex[coord] = result;
}
