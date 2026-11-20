#include "ModelLoader.h"
#include "Renderer/Renderer.h"
#include "Vertex.h"
#include "Material.h"
#include "volk.h"
#include <filesystem>
#include <stdexcept>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/mesh.h>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

void ModelLoader::loadModel(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices, std::vector<Material> &materials, Renderer &renderer) {
    Assimp::Importer importer{};
    const char *path{PROJECT_ROOT_DIR "models/scene.gltf"};
    const aiScene *scene{importer.ReadFile(
        path,
        aiProcess_Triangulate |
        aiProcess_PreTransformVertices |  // bake node transforms (the helmet's node rotates it upright)
        aiProcess_JoinIdenticalVertices |
        aiProcess_FlipUVs
    )};

    if (scene == nullptr || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) != 0 || !scene->mRootNode) {
        std::string error{"Assimp error: "};
        error += importer.GetErrorString();
        error += "\n";
        throw std::runtime_error(error);
    }

    std::filesystem::path modelDir{std::filesystem::path(path).parent_path()};
    std::unordered_map<std::string, uint32_t> texturePathToSlot{};
    std::unordered_map<uint32_t, uint32_t> assimpMatToCompact{};

    for (uint32_t i = 0; i < scene->mNumMeshes; ++i) {
        const aiMesh *mesh{scene->mMeshes[i]};

        uint32_t assimpMatIndex{mesh->mMaterialIndex};
        uint32_t compactMatIndex{};

        auto matIt{assimpMatToCompact.find(assimpMatIndex)};
        if (matIt != assimpMatToCompact.end()) {
            compactMatIndex = matIt->second;
        } else {
            const aiMaterial *material{scene->mMaterials[assimpMatIndex]};

            uint32_t albedoIndex{0xFFFFFFFF};
            aiString albedoPath{};
            if (material->GetTexture(aiTextureType_BASE_COLOR, 0, &albedoPath) != AI_SUCCESS) {
                material->GetTexture(aiTextureType_DIFFUSE, 0, &albedoPath);
            }
            if (albedoPath.length > 0) {
                std::string fullPath{(modelDir / albedoPath.C_Str()).string()};
                auto it{texturePathToSlot.find(fullPath)};
                if (it != texturePathToSlot.end()) {
                    albedoIndex = it->second;
                } else {
                    albedoIndex = renderer.uploadTexture(fullPath, VK_FORMAT_R8G8B8A8_SRGB);
                    texturePathToSlot.emplace(fullPath, albedoIndex);
                }
            }

            uint32_t normalIndex{0xFFFFFFFF};
            aiString normalPath{};
            if (material->GetTexture(aiTextureType_NORMALS, 0, &normalPath) == AI_SUCCESS) {
                std::string fullPath{(modelDir / normalPath.C_Str()).string()};
                auto it{texturePathToSlot.find(fullPath)};
                if (it != texturePathToSlot.end()) {
                    normalIndex = it->second;
                } else {
                    normalIndex = renderer.uploadTexture(fullPath, VK_FORMAT_R8G8B8A8_UNORM);
                    texturePathToSlot.emplace(fullPath, normalIndex);
                }
            }

            compactMatIndex = static_cast<uint32_t>(materials.size());
            materials.push_back(Material{.albedoIndex = albedoIndex, .normalIndex = normalIndex});
            assimpMatToCompact.emplace(assimpMatIndex, compactMatIndex);
        }

        uint32_t vertexOffset{static_cast<uint32_t>(vertices.size())};
        for (uint32_t j = 0; j < mesh->mNumVertices; ++j) {
            Vertex vertex{};
            vertex.position = {mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z};

            if (mesh->HasNormals()) {
                vertex.normal = {mesh->mNormals[j].x, mesh->mNormals[j].y, mesh->mNormals[j].z};
            }
            if (mesh->HasTextureCoords(0)) {
                vertex.texCoords = {mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y};
            }

            vertex.materialIndex = compactMatIndex;
            vertices.push_back(vertex);
        }

        for (uint32_t j = 0; j < mesh->mNumFaces; ++j) {
            const aiFace &face{mesh->mFaces[j]};
            for (uint32_t k = 0; k < face.mNumIndices; ++k) {
                indices.push_back(vertexOffset + face.mIndices[k]);
            }
        }
    }
}
