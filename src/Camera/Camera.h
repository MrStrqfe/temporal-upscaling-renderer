#pragma once
#include <glm/glm.hpp>

class Window;

class Camera {
public:
    void update();
    void processKeyboard(Window &window, float deltaTime);
    void processMouseMovement(float xPos, float yPos);

    glm::mat4 getCurrentVP() const;
    glm::mat4 getPreviousVP() const;
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    glm::mat4 getProjectionMatrixNoJitter() const;
    glm::vec2 getJitterOffset() const;
    glm::vec3 getPosition() const;
    void setAspectRatio(float aspectRatio);
    void incrementFrameIndex();

private:
    // position & orientation
    glm::vec3 m_position  {0.0f, 0.0f, 3.0f};
    glm::vec3 m_front     {0.0f, 0.0f, -1.0f};
    glm::vec3 m_up        {0.0f, 1.0f, 0.0f};
    float     m_yaw       {-90.0f};
    float     m_pitch     {0.0f};

    // mouse state
    float m_lastMouseX{};
    float m_lastMouseY{};
    bool  m_firstMouse{true};

    // projection
    float m_fov        {45.0f};
    float m_aspectRatio{16.0f / 9.0f};
    float m_nearPlane  {0.1f};
    float m_farPlane   {100.0f};

    // temporal state
    glm::mat4 m_currentVP {1.0f};
    glm::mat4 m_previousVP{1.0f};
    uint32_t  m_frameIndex{0};

    // jitter helpers
    float halton(uint32_t index, uint32_t base) const;
    glm::vec2 computeJitter() const;
};
