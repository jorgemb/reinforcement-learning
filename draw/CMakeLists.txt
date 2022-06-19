cmake_minimum_required (VERSION 3.16)

project ("draw")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# LibDRAW
set(LIBDRAW_SOURCES
        src/grid.cpp)
set(LIBDRAW_HEADERS
        include/draw/grid.h)
IF(WIN32)
    add_library(draw STATIC ${LIBDRAW_SOURCES} ${LIBDRAW_HEADERS})
else()
    add_library(draw SHARED ${LIBDRAW_SOURCES} ${LIBDRAW_HEADERS})
endif()
target_include_directories(draw PUBLIC include)
target_link_libraries(draw PUBLIC sfml-system sfml-graphics sfml-window mdp)
set_target_properties(draw PROPERTIES
        PUBLIC_HEADER "${LIBDRAW_HEADERS}"
        )

install(
        TARGETS draw
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include/draw/
)

# Executables
add_executable(draw-gridworld draw_gridworld.cpp)
target_link_libraries(draw-gridworld PRIVATE draw fmt::fmt)