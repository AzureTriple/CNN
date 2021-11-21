#pragma once
#include <sstream>
#include <stdexcept>
#include <type_traits>

template<typename...Args>
void err(Args&&...args) {
    throw std::logic_error((std::ostringstream() << ... << std::forward<Args>(args)).str().c_str());
}