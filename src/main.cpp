#include "Window/Window.h"
#include "Input/Input.h"
#include <iostream>
#include <exception>

int main() {
    try {
        Window window{};
        Input input{};
        window.initWindow();

        while (!window.shouldClose()) {
            window.pollEvents();
            input.processUserInput(window);
        }

        window.cleanup();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}
