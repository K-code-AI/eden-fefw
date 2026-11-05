function(eden_profile_link_options)
    if(MSVC AND TARGET yuzu)
        if(EDEN_LOCAL_PROFILE_STAGE STREQUAL "GENERATE")
            target_link_options(yuzu PRIVATE /LTCG
                /FASTGENPROFILE:NOEXACT,COUNTER64,PGD=../pgo-crowd-20261104/eden.pgd)
        elseif(EDEN_LOCAL_PROFILE_STAGE STREQUAL "USE")
            target_link_options(yuzu PRIVATE /LTCG
                /USEPROFILE:PGD=../pgo-crowd-20261104/eden.pgd)
        elseif(EDEN_LOCAL_PROFILE_STAGE STREQUAL "AGGRESSIVE")
            target_link_options(yuzu PRIVATE /LTCG
                /USEPROFILE:AGGRESSIVE,PGD=../pgo-crowd-20261104/eden.pgd)
        else()
            message(FATAL_ERROR "Unknown local profile stage")
        endif()
    endif()
    if(MSVC AND TARGET video_core AND EDEN_LOCAL_RENDER_ISA)
        if(NOT EDEN_LOCAL_RENDER_ISA MATCHES "^(AVX2|AVX512)$")
            message(FATAL_ERROR "Unknown local renderer ISA")
        endif()
        foreach(unit IN ITEMS
                renderer_vulkan/vk_graphics_pipeline.cpp
                renderer_vulkan/vk_texture_cache_base.cpp
                renderer_vulkan/vk_buffer_cache.cpp
                renderer_vulkan/vk_rasterizer.cpp
                renderer_vulkan/vk_compute_pipeline.cpp
        )
            set_property(SOURCE "${CMAKE_SOURCE_DIR}/src/video_core/${unit}"
                TARGET_DIRECTORY video_core APPEND PROPERTY COMPILE_OPTIONS
                "/arch:${EDEN_LOCAL_RENDER_ISA}" /fp:precise)
        endforeach()
    endif()
endfunction()
cmake_language(DEFER CALL eden_profile_link_options)
