#version {{version}}
// Automatically generated from files in pathfinder/shaders/. Do not edit!












precision highp float;

uniform sampler2D uStencilTexture;
uniform sampler2D uPaintTexture;

in vec2 vColorTexCoord;
in vec2 vMaskTexCoord;

out vec4 oFragColor;

void main(){
    float coverage = texture(uStencilTexture, vMaskTexCoord). r;
    vec4 color = texture(uPaintTexture, vColorTexCoord);
    color . a *= coverage;
    color . rgb *= color . a;
    oFragColor = color;
}

