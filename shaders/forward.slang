[[vk::binding(0, 0)]]
Sampler2D g_textures[];

struct Vertex {
    float3 position;
    float3 normal;
    float2 texCoord;
    uint32_t materialIndex;
};

struct Material {
    uint albedoIndex;
    uint normalIndex;
};

struct PushConstants {
    Material *materialBuffer;
    Vertex *vertexBuffer;
    float4x4 model;
    float4x4 view;
    float4x4 projection;
    float4x4 prevViewProjection;
    float2 jitterOffset;
};

struct VertexOutput {
    float4 position : SV_Position;
    float3 normal;
    float2 texCoord;
    float3 lightDir;
    float3 viewDir;
    nointerpolation uint materialIndex;
    float4 currentClipPos : CLIP_POS;
    float4 previousClipPos : PREV_CLIP_POS;
};

[[vk::push_constant]]
PushConstants pushConstants;

const static uint g_invalidTextureIndex = 0xFFFFFFFF;
const static float g_ambientStrength = 0.1;
const static float3 g_lightColor = {1.0, 1.0, 1.0};
const static float4 g_lightPos = {-2.5, 4.0, 5.0, 1.0};
const static float g_shininess = 32.0;
const static float g_specularStrength = 0.5;

[shader("vertex")]
VertexOutput vertexMain(uint vertexIndex: SV_VertexID) {
    Vertex vertex = pushConstants.vertexBuffer[vertexIndex];
    float4x4 model = pushConstants.model;
    float4x4 view = pushConstants.view;
    float4x4 projection = pushConstants.projection;

    float4 worldPos = mul(model, float4(vertex.position, 1.0));
    float4 viewPos = mul(view, worldPos);

    VertexOutput output;
    output.position = mul(projection, viewPos);
    output.normal   = mul((float3x3)model, vertex.normal);
    output.texCoord = vertex.texCoord;
    output.lightDir = g_lightPos.xyz - worldPos.xyz;
    output.viewDir  = -viewPos.xyz;
    output.materialIndex = vertex.materialIndex;
    output.currentClipPos = output.position;
    output.previousClipPos = mul(pushConstants.prevViewProjection, mul(model, float4(vertex.position, 1.0)));
    return output;
}

struct FragmentOutput {
    float4 color    : SV_Target0;
    float2 velocity : SV_Target1;
};

[shader("fragment")]
FragmentOutput fragmentMain(VertexOutput vertex) {
    Material material = pushConstants.materialBuffer[vertex.materialIndex];

    float4 baseColor = float4(1.0, 1.0, 1.0, 1.0);
    if (material.albedoIndex != g_invalidTextureIndex) {
        baseColor = g_textures[NonUniformResourceIndex(material.albedoIndex)].Sample(vertex.texCoord);
    }

    if (baseColor.a < 0.1) {
        discard;
    }

    float3 albedo = baseColor.rgb;

    float3 normal = normalize(vertex.normal);
    float3 lightDir = normalize(vertex.lightDir);
    float3 viewDir = normalize(vertex.viewDir);

    float3 ambient = g_ambientStrength * g_lightColor * albedo;

    float diffuseFactor = max(dot(normal, lightDir), 0.0);
    float3 diffuse = diffuseFactor * g_lightColor * albedo;

    float3 halfwayDir = normalize(lightDir + viewDir);
    float specularFactor = pow(max(dot(normal, halfwayDir), 0.0), g_shininess);
    float3 specular = g_specularStrength * specularFactor * g_lightColor;

    float3 result = ambient + specular + diffuse;

    FragmentOutput output;
    output.color = float4(result, 1.0);

    // Compute screen-space motion vector (NDC delta, scaled to [-0.5, 0.5])
    float2 currentNDC = vertex.currentClipPos.xy / vertex.currentClipPos.w;
    float2 previousNDC = vertex.previousClipPos.xy / vertex.previousClipPos.w;
    output.velocity = (currentNDC - previousNDC) * 0.5;

    return output;
}
