#pragma once
#include <vector>
#include <cstdint>
struct Vertex;
struct Material;
class Renderer;

class ModelLoader {
public:
    void loadModel(
        std::vector<Vertex> &vertices,
        std::vector<uint32_t> &indices,
        std::vector<Material> &materials,
        Renderer &renderer
    );
};
