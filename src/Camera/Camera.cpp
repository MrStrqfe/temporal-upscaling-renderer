#include "Camera.h"
#include "Window/Window.h"
#include "Config.h"
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <cmath>
#include <algorithm>
#include <GLFW/glfw3.h>

void Camera::update() {
    m_previousVP = m_currentVP;

    glm::mat4 view{glm::lookAt(m_position, m_position + m_front, m_up)};
    glm::mat4 proj{getProjectionMatrix()};

    m_currentVP = proj * view;
}

void Camera::processKeyboard(Window &window, float deltaTime) {
    const float cameraSpeed{2.5f * deltaTime};
    if (window.isKeyPressed(GLFW_KEY_W))
        m_position += cameraSpeed * m_front;
    if (window.isKeyPressed(GLFW_KEY_S))
        m_position -= cameraSpeed * m_front;
    if (window.isKeyPressed(GLFW_KEY_A))
        m_position -= glm::normalize(glm::cross(m_front, m_up)) * cameraSpeed;
    if (window.isKeyPressed(GLFW_KEY_D))
        m_position += glm::normalize(glm::cross(m_front, m_up)) * cameraSpeed;
}

void Camera::processMouseMovement(float xPos, float yPos) {
    if (m_firstMouse) {
        m_lastMouseX = xPos;
        m_lastMouseY = yPos;
        m_firstMouse = false;
    }

    float xOffset{xPos - m_lastMouseX};
    float yOffset{m_lastMouseY - yPos};
    m_lastMouseX = xPos;
    m_lastMouseY = yPos;

    constexpr float sensitivity{0.05f};
    xOffset *= sensitivity;
    yOffset *= sensitivity;

    m_yaw += xOffset;
    m_pitch += yOffset;
    m_pitch = std::clamp(m_pitch, -89.0f, 89.0f);

    glm::vec3 front{
        cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch)),
        sin(glm::radians(m_pitch)),
        sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch))
    };
    m_front = glm::normalize(front);
}

glm::mat4 Camera::getCurrentVP() const {
    return m_currentVP;
}

glm::mat4 Camera::getPreviousVP() const {
    return m_previousVP;
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(m_position, m_position + m_front, m_up);
}

glm::mat4 Camera::getProjectionMatrix() const {
    glm::mat4 proj{getProjectionMatrixNoJitter()};

    // Apply sub-pixel jitter for TAA/upscaling
    glm::vec2 jitter{computeJitter()};
    // Convert pixel jitter to NDC offset
    float internalWidth  {Config::width  * Config::upscaleRatio};
    float internalHeight {Config::height * Config::upscaleRatio};
    proj[2][0] += jitter.x / internalWidth;
    proj[2][1] += jitter.y / internalHeight;

    return proj;
}

glm::mat4 Camera::getProjectionMatrixNoJitter() const {
    glm::mat4 proj{glm::perspective(
        glm::radians(m_fov),
        m_aspectRatio,
        m_nearPlane,
        m_farPlane
    )};
    proj[1][1] *= -1;  // flip Y for Vulkan
    return proj;
}

glm::vec2 Camera::getJitterOffset() const {
    return computeJitter();
}

glm::vec3 Camera::getPosition() const {
    return m_position;
}

void Camera::setAspectRatio(float aspectRatio) {
    m_aspectRatio = aspectRatio;
}

void Camera::incrementFrameIndex() {
    m_frameIndex++;
}

float Camera::halton(uint32_t index, uint32_t base) const {
    float result{0.0f};
    float fraction{1.0f};
    uint32_t i{index};
    while (i > 0) {
        fraction /= static_cast<float>(base);
        result += fraction * static_cast<float>(i % base);
        i /= base;
    }
    return result;
}

glm::vec2 Camera::computeJitter() const {
    // Halton sequence (base 2, 3) centered around 0
    uint32_t idx{(m_frameIndex % Config::jitterSequenceLen) + 1};
    return {
        halton(idx, 2) - 0.5f,
        halton(idx, 3) - 0.5f
    };
}
