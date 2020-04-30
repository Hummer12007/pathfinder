#version 330

// pathfinder/shaders/tile.fs.glsl
//
// Copyright © 2020 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//      Mask UV 0         Mask UV 1
//          +                 +
//          |                 |
//    +-----v-----+     +-----v-----+
//    |           | MIN |           |
//    |  Mask  0  +----->  Mask  1  +------+
//    |           |     |           |      |
//    +-----------+     +-----------+      v       +-------------+
//                                       Apply     |             |       GPU
//                                       Mask +---->  Composite  +---->Blender
//                                         ^       |             |
//    +-----------+     +-----------+      |       +-------------+
//    |           |     |           |      |
//    |  Color 0  +----->  Color 1  +------+
//    |  Filter   |  ×  |           |
//    |           |     |           |
//    +-----^-----+     +-----^-----+
//          |                 |
//          +                 +
//     Color UV 0        Color UV 1

#extension GL_GOOGLE_include_directive : enable

precision highp float;
precision highp sampler2D;

uniform sampler2D uColorTexture0;
uniform sampler2D uMaskTexture0;
uniform sampler2D uDestTexture;
uniform sampler2D uGammaLUT;
uniform vec2 uColorTextureSize0;
uniform vec2 uMaskTextureSize0;
uniform vec4 uFilterParams0;
uniform vec4 uFilterParams1;
uniform vec4 uFilterParams2;
uniform vec2 uFramebufferSize;
uniform int uCtrl;

in vec3 vMaskTexCoord0;
in vec2 vColorTexCoord0;
in vec4 vBaseColor;
in float vTileCtrl;

out vec4 oFragColor;

#include "tile.inc.glsl"

vec4 calculateColor(int tileCtrl, int ctrl) {
    // Sample mask.
    int maskCtrl0 = (tileCtrl >> TILE_CTRL_MASK_0_SHIFT) & TILE_CTRL_MASK_MASK;
    float maskAlpha = 1.0;
    maskAlpha = sampleMask(maskAlpha, uMaskTexture0, uMaskTextureSize0, vMaskTexCoord0, maskCtrl0);
    return calculateColorWithMaskAlpha(maskAlpha,
                                       vBaseColor,
                                       vColorTexCoord0,
                                       gl_FragCoord.xy,
                                       ctrl);
}

// Entry point
//
// TODO(pcwalton): Generate this dynamically.

void main() {
    oFragColor = calculateColor(int(vTileCtrl), uCtrl);
}
