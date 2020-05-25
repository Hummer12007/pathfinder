#version 450

// pathfinder/shaders/bin.vs.glsl
//
// Copyright © 2020 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

precision highp float;

#ifdef GL_ES
precision highp sampler2D;
#endif

uniform ivec2 uFramebufferSize;

struct Segment {
    vec4 line;
    uvec4 pathIndex;
};

layout(std430, binding = 4) buffer bSegments {
    restrict readonly Segment iSegments[];
};

out vec2 vFrom;
out vec2 vTo;
flat out uint vPathIndex;

void main() {
    uint segmentIndex = gl_VertexID / 6u;

    vec2 tessCoord;
    switch (gl_VertexID % 6) {
    case 0:         tessCoord = vec2(0.0, 0.0); break;
    case 1: case 3: tessCoord = vec2(1.0, 0.0); break;
    case 2: case 5: tessCoord = vec2(0.0, 1.0); break;
    case 4:         tessCoord = vec2(1.0, 1.0); break;
    }

    vec4 line = iSegments[segmentIndex].line;
    uint pathIndex = iSegments[segmentIndex].pathIndex.x;

    vec2 from = line.xy / vec2(16.0), to = line.zw / vec2(16.0);
    vec2 vector = normalize(to - from) * vec2(0.5);
    vec2 normal = vec2(-vector.y, vector.x);
    vec2 tilePosition = mix(from - vector, to + vector, tessCoord.y) +
        mix(-normal, normal, tessCoord.x);

    vFrom = from;
    vTo = to;
    vPathIndex = pathIndex;

    gl_Position = vec4(mix(vec2(-1.0), vec2(1.0), tilePosition / uFramebufferSize), 0.0, 1.0);
}
