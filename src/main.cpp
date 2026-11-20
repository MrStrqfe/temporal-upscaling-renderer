#include "App/App.h"
#include <iostream>
#include <exception>

int main() {
    try {
        App app{};
        app.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}
