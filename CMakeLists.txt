cmake_minimum_required(VERSION 3.10)
project(MINDFIELD LANGUAGES CUDA CXX VERSION 0.1)

# Standard di C++ usato
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    set(CMAKE_CXX_STANDARD 17)
endif()

# Standardi di CUDA usato
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    set(CMAKE_CUDA_STANDARD 11)
endif()

# =====================================================================
# Eseguibile - Mindfield

set(MINDFIELD_CONFIG_IN  "${PROJECT_SOURCE_DIR}/config/config.hpp.in")
set(MINDFIELD_CONFIG_OUT "${PROJECT_BINARY_DIR}/config.hpp")

# Genera header di configurazione
configure_file(${MINDFIELD_CONFIG_IN} ${MINDFIELD_CONFIG_OUT} @ONLY)

add_executable(Mindfield source/main.cu)
target_include_directories(Mindfield PUBLIC
    "${PROJECT_BINARY_DIR}"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/extern"
)

target_compile_options(Mindfield PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_35,code=sm_35>)

# =====================================================================
# Doxygen

option(BUILD_DOXYGEN_DOCS "Build documentation" ON)

find_package(Doxygen)
if(DOXYGEN_FOUND)

    set(DOXYGEN_CONFIG_IN  "${PROJECT_SOURCE_DIR}/config/Doxyfile.in")
    set(DOXYGEN_CONFIG_OUT "${PROJECT_SOURCE_DIR}/docs/Doxyfile")

    # Genera configurazione per Doxygen
    configure_file(${DOXYGEN_CONFIG_IN} ${DOXYGEN_CONFIG_OUT} @ONLY)

    message("Avvio generazione documentazione Doxygen...")
    add_custom_target(Doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )

endif(DOXYGEN_FOUND)