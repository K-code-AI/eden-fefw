// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/surface.h"
#include "video_core/textures/texture.h"

namespace VideoCommon {

// Resolve a shader format query, including an unbound texture descriptor.
VideoCore::Surface::PixelFormat PixelFormatFromTextureInfo(
    const Tegra::Texture::TICEntry& entry) noexcept;

VideoCore::Surface::PixelFormat PixelFormatFromTextureInfo(
    Tegra::Texture::TextureFormat format, Tegra::Texture::ComponentType red_component,
    Tegra::Texture::ComponentType green_component, Tegra::Texture::ComponentType blue_component,
    Tegra::Texture::ComponentType alpha_component, bool is_srgb) noexcept;

} // namespace VideoCommon
