#include "App.h"

void App::run() {
    m_window.initWindow();
    m_window.captureCursor();

    // Set up mouse callback — store Camera pointer in GLFW user pointer
    glfwSetWindowUserPointer(m_window.getWindow(), &m_camera);
    glfwSetCursorPosCallback(m_window.getWindow(), [](GLFWwindow *window, double xPos, double yPos) {
        auto *camera{static_cast<Camera *>(glfwGetWindowUserPointer(window))};
        camera->processMouseMovement(static_cast<float>(xPos), static_cast<float>(yPos));
    });

    m_renderer.initVulkan(m_window);
    m_camera.setAspectRatio(
        static_cast<float>(m_renderer.getSwapchainExtent().width) /
        static_cast<float>(m_renderer.getSwapchainExtent().height)
    );

    m_motionVectors.init(m_renderer);
    m_renderer.setVelocityTarget(m_motionVectors.getImageView(), m_motionVectors.getImage());

    m_modelLoader.loadModel(m_vertices, m_indices, m_materials, m_renderer);
    m_renderer.uploadModel(m_vertices, m_indices);
    m_renderer.uploadMaterials(m_materials);

    mainLoop();
    cleanup();
}

void App::mainLoop() {
    while (!m_window.shouldClose()) {
        float currentFrame{m_window.getTime()};
        m_deltaTime = currentFrame - m_lastFrame;
        m_lastFrame = currentFrame;

        m_window.pollEvents();
        m_input.processUserInput(m_window);
        m_camera.processKeyboard(m_window, m_deltaTime);
        m_camera.update();

        m_renderer.drawFrame(m_window, m_camera);

        m_camera.incrementFrameIndex();
    }
}

void App::cleanup() {
    m_motionVectors.cleanup(m_renderer);
    m_renderer.cleanup();
    m_window.cleanup();
}
