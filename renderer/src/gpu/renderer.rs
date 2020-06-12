// pathfinder/renderer/src/gpu/renderer.rs
//
// Copyright © 2020 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::gpu::d3d11::renderer::RendererD3D11;
use crate::gpu::debug::DebugUIPresenter;
use crate::gpu::mem::{BufferID, BufferTag, FramebufferID, FramebufferTag};
use crate::gpu::mem::{GPUMemoryAllocator, PatternTexturePage, TextureID, TextureTag};
use crate::gpu::options::{DestFramebuffer, RendererLevel, RendererOptions};
use crate::gpu::perf::{PendingTimer, RenderStats, RenderTime, TimerFuture, TimerQueryCache};
use crate::gpu::shaders::{BlitProgram, BlitVertexArray, ClearProgram};
use crate::gpu::shaders::{ClearVertexArray};
use crate::gpu::shaders::{CopyTileProgram, MAX_FILLS_PER_BATCH};
use crate::gpu::shaders::{ProgramsCore, ProgramsD3D9, ReprojectionProgram, ReprojectionVertexArray};
use crate::gpu::shaders::{StencilProgram, StencilVertexArray};
use crate::gpu::shaders::{TileProgramCommon, VertexArraysCore};
use crate::gpu_data::{Fill};
use crate::gpu_data::{RenderCommand};
use crate::gpu_data::{TextureLocation, TextureMetadataEntry, TexturePageDescriptor};
use crate::gpu_data::{TexturePageId, TileBatchTexture, TileObjectPrimitive};
use crate::options::BoundingQuad;
use crate::paint::PaintCompositeOp;
use crate::tile_map::DenseTileMap;
use crate::tiles::{TILE_HEIGHT, TILE_WIDTH};
use byte_slice_cast::AsByteSlice;
use half::f16;
use pathfinder_color::{self as color, ColorF, ColorU};
use pathfinder_content::effects::{BlendMode, BlurDirection, DefringingKernel};
use pathfinder_content::effects::{Filter, PatternFilter};
use pathfinder_content::render_target::RenderTargetId;
use pathfinder_geometry::line_segment::LineSegment2F;
use pathfinder_geometry::rect::{RectF, RectI};
use pathfinder_geometry::transform3d::Transform4F;
use pathfinder_geometry::util;
use pathfinder_geometry::vector::{Vector2F, Vector2I, Vector4F, vec2f, vec2i};
use pathfinder_gpu::{BlendFactor, BlendState, BufferData, BufferTarget};
use pathfinder_gpu::{ClearOps, DepthFunc, DepthState, Device};
use pathfinder_gpu::{Primitive, RenderOptions, RenderState, RenderTarget};
use pathfinder_gpu::{StencilFunc, StencilState, TextureBinding, TextureDataRef, TextureFormat};
use pathfinder_gpu::{UniformBinding, UniformData};
use pathfinder_resources::ResourceLoader;
use pathfinder_simd::default::{F32x2, F32x4, I32x2};
use std::collections::VecDeque;
use std::f32;
use std::mem;
use std::time::Duration;
use std::u32;

static QUAD_VERTEX_POSITIONS: [u16; 8] = [0, 0, 1, 0, 1, 1, 0, 1];
static QUAD_VERTEX_INDICES: [u32; 6] = [0, 1, 3, 1, 2, 3];

pub(crate) const MASK_TILES_ACROSS: u32 = 256;
pub(crate) const MASK_TILES_DOWN: u32 = 256;

// 1.0 / sqrt(2*pi)
const SQRT_2_PI_INV: f32 = 0.3989422804014327;

const TEXTURE_METADATA_ENTRIES_PER_ROW: i32 = 128;
const TEXTURE_METADATA_TEXTURE_WIDTH:   i32 = TEXTURE_METADATA_ENTRIES_PER_ROW * 4;
const TEXTURE_METADATA_TEXTURE_HEIGHT:  i32 = 65536 / TEXTURE_METADATA_ENTRIES_PER_ROW;

// FIXME(pcwalton): Shrink this again!
const MASK_FRAMEBUFFER_WIDTH:  i32 = TILE_WIDTH as i32      * MASK_TILES_ACROSS as i32;
const MASK_FRAMEBUFFER_HEIGHT: i32 = TILE_HEIGHT as i32 / 4 * MASK_TILES_DOWN as i32;

const COMBINER_CTRL_COLOR_COMBINE_SRC_IN: i32 =     0x1;
const COMBINER_CTRL_COLOR_COMBINE_DEST_IN: i32 =    0x2;

const COMBINER_CTRL_FILTER_RADIAL_GRADIENT: i32 =   0x1;
const COMBINER_CTRL_FILTER_TEXT: i32 =              0x2;
const COMBINER_CTRL_FILTER_BLUR: i32 =              0x3;

const COMBINER_CTRL_COMPOSITE_NORMAL: i32 =         0x0;
const COMBINER_CTRL_COMPOSITE_MULTIPLY: i32 =       0x1;
const COMBINER_CTRL_COMPOSITE_SCREEN: i32 =         0x2;
const COMBINER_CTRL_COMPOSITE_OVERLAY: i32 =        0x3;
const COMBINER_CTRL_COMPOSITE_DARKEN: i32 =         0x4;
const COMBINER_CTRL_COMPOSITE_LIGHTEN: i32 =        0x5;
const COMBINER_CTRL_COMPOSITE_COLOR_DODGE: i32 =    0x6;
const COMBINER_CTRL_COMPOSITE_COLOR_BURN: i32 =     0x7;
const COMBINER_CTRL_COMPOSITE_HARD_LIGHT: i32 =     0x8;
const COMBINER_CTRL_COMPOSITE_SOFT_LIGHT: i32 =     0x9;
const COMBINER_CTRL_COMPOSITE_DIFFERENCE: i32 =     0xa;
const COMBINER_CTRL_COMPOSITE_EXCLUSION: i32 =      0xb;
const COMBINER_CTRL_COMPOSITE_HUE: i32 =            0xc;
const COMBINER_CTRL_COMPOSITE_SATURATION: i32 =     0xd;
const COMBINER_CTRL_COMPOSITE_COLOR: i32 =          0xe;
const COMBINER_CTRL_COMPOSITE_LUMINOSITY: i32 =     0xf;

const COMBINER_CTRL_COLOR_FILTER_SHIFT: i32 =       4;
const COMBINER_CTRL_COLOR_COMBINE_SHIFT: i32 =      6;
const COMBINER_CTRL_COMPOSITE_SHIFT: i32 =          8;

pub struct Renderer<D> where D: Device {
    pub(crate) core: RendererCore<D>,
    level_impl: RendererLevelImpl<D>,

    // Core data
    blit_program: BlitProgram<D>,
    clear_program: ClearProgram<D>,
    tile_copy_program: CopyTileProgram<D>,
    d3d9_programs: Option<ProgramsD3D9<D>>,
    stencil_program: StencilProgram<D>,
    reprojection_program: ReprojectionProgram<D>,

    // Frames
    frame: Frame<D>,

    // Debug
    current_cpu_build_time: Option<Duration>,
    pending_timers: VecDeque<PendingTimer<D>>,
    debug_ui_presenter: DebugUIPresenter<D>,
}

enum RendererLevelImpl<D> where D: Device {
    D3D11(RendererD3D11<D>),
}

pub(crate) struct RendererCore<D> where D: Device {
    pub(crate) device: D,
    pub(crate) allocator: GPUMemoryAllocator<D>,
    pub(crate) options: RendererOptions,
    pub(crate) stats: RenderStats,
    pub(crate) current_timer: Option<PendingTimer<D>>,
    pub(crate) timer_query_cache: TimerQueryCache<D>,
    pub(crate) dest_framebuffer: DestFramebuffer<D>,
    pub(crate) renderer_flags: RendererFlags,

    // Core shaders
    pub(crate) programs: ProgramsCore<D>,
    pub(crate) vertex_arrays: VertexArraysCore<D>,

    // Read-only static core resources
    pub(crate) quad_vertex_positions_buffer_id: BufferID,
    pub(crate) quad_vertex_indices_buffer_id: BufferID,
    pub(crate) area_lut_texture_id: TextureID,
    pub(crate) gamma_lut_texture_id: TextureID,

    // Read-write static core resources
    intermediate_dest_framebuffer_id: FramebufferID,
    pub(crate) texture_metadata_texture_id: TextureID,

    // Dynamic resources and associated metadata
    render_targets: Vec<RenderTargetInfo>,
    pub(crate) render_target_stack: Vec<RenderTargetId>,
    pub(crate) pattern_texture_pages: Vec<Option<PatternTexturePage>>,
    pub(crate) mask_storage: Option<MaskStorage>,
    pub(crate) alpha_tile_count: u32,
    pub(crate) framebuffer_flags: FramebufferFlags,
}

struct Frame<D> where D: Device {
    blit_vertex_array: BlitVertexArray<D>,
    clear_vertex_array: ClearVertexArray<D>,
    // Maps tile batch IDs to tile vertex storage IDs.
    quads_vertex_indices_buffer_id: Option<BufferID>,
    quads_vertex_indices_length: usize,
    buffered_fills: Vec<Fill>,
    pending_fills: Vec<Fill>,
    // Temporary place that we copy tiles to in order to perform clips, allocated lazily.
    //
    // TODO(pcwalton): This should be sparse, not dense.
    mask_temp_framebuffer: Option<FramebufferID>,
    stencil_vertex_array: StencilVertexArray<D>,
    reprojection_vertex_array: ReprojectionVertexArray<D>,
    dest_blend_framebuffer_id: FramebufferID,
}

pub(crate) struct MaskStorage {
    pub(crate) framebuffer_id: FramebufferID,
    allocated_page_count: u32,
}

impl<D> Renderer<D> where D: Device {
    pub fn new(device: D,
               resources: &dyn ResourceLoader,
               dest_framebuffer: DestFramebuffer<D>,
               options: RendererOptions)
               -> Renderer<D> {
        let mut allocator = GPUMemoryAllocator::new();

        let quad_vertex_positions_buffer_id =
            allocator.allocate_buffer::<u16>(&device,
                                             QUAD_VERTEX_POSITIONS.len() as u64,
                                             BufferTag::QuadVertexPositions);
        device.upload_to_buffer(allocator.get_buffer(quad_vertex_positions_buffer_id),
                                0,
                                &QUAD_VERTEX_POSITIONS,
                                BufferTarget::Vertex);
        let quad_vertex_indices_buffer_id =
            allocator.allocate_buffer::<u32>(&device,
                                             QUAD_VERTEX_INDICES.len() as u64,
                                             BufferTag::QuadVertexIndices);
        device.upload_to_buffer(allocator.get_buffer(quad_vertex_indices_buffer_id),
                                0,
                                &QUAD_VERTEX_INDICES,
                                BufferTarget::Index);

        let area_lut_texture_id = allocator.allocate_texture(&device,
                                                             Vector2I::splat(256),
                                                             TextureFormat::RGBA8,
                                                             TextureTag::AreaLUT);
        let gamma_lut_texture_id = allocator.allocate_texture(&device,
                                                              vec2i(256, 8),
                                                              TextureFormat::R8,
                                                              TextureTag::GammaLUT);
        device.upload_png_to_texture(resources,
                                     "area-lut",
                                     allocator.get_texture(area_lut_texture_id),
                                     TextureFormat::RGBA8);
        device.upload_png_to_texture(resources,
                                     "gamma-lut",
                                     allocator.get_texture(gamma_lut_texture_id),
                                     TextureFormat::R8);

        let window_size = dest_framebuffer.window_size(&device);

        let intermediate_dest_framebuffer_id =
            allocator.allocate_framebuffer(&device,
                                           window_size,
                                           TextureFormat::RGBA8,
                                           FramebufferTag::IntermediateDest);

        let texture_metadata_texture_size = vec2i(TEXTURE_METADATA_TEXTURE_WIDTH,
                                                  TEXTURE_METADATA_TEXTURE_HEIGHT);
        let texture_metadata_texture_id = allocator.allocate_texture(&device,
                                                                     texture_metadata_texture_size,
                                                                     TextureFormat::RGBA16F,
                                                                     TextureTag::TextureMetadata);

        let core_programs = ProgramsCore::new(&device, resources);
        let core_vertex_arrays =
             VertexArraysCore::new(&device,
                                   &core_programs,
                                   allocator.get_buffer(quad_vertex_positions_buffer_id),
                                   allocator.get_buffer(quad_vertex_indices_buffer_id));

        let mut core = RendererCore {
            device,
            allocator,
            options,
            stats: RenderStats::default(),
            current_timer: None,
            timer_query_cache: TimerQueryCache::new(),
            dest_framebuffer,
            renderer_flags: RendererFlags::empty(),

            programs: core_programs,
            vertex_arrays: core_vertex_arrays,

            quad_vertex_positions_buffer_id,
            quad_vertex_indices_buffer_id,
            area_lut_texture_id,
            gamma_lut_texture_id,

            intermediate_dest_framebuffer_id,

            texture_metadata_texture_id,
            render_targets: vec![],
            render_target_stack: vec![],
            pattern_texture_pages: vec![],
            mask_storage: None,
            alpha_tile_count: 0,
            framebuffer_flags: FramebufferFlags::empty(),
        };

        let level_impl = RendererLevelImpl::D3D11(RendererD3D11::new(&mut core, resources));

        let blit_program = BlitProgram::new(&core.device, resources);
        let clear_program = ClearProgram::new(&core.device, resources);
        let tile_copy_program = CopyTileProgram::new(&core.device, resources);
        let stencil_program = StencilProgram::new(&core.device, resources);
        let reprojection_program = ReprojectionProgram::new(&core.device, resources);

        let d3d9_programs = match core.options.level {
            RendererLevel::D3D9 => Some(ProgramsD3D9::new(&core.device, resources)),
            RendererLevel::D3D11 => None,
        };

        let debug_ui_presenter = DebugUIPresenter::new(&core.device,
                                                       resources,
                                                       window_size,
                                                       core.options.level);

        let frame = Frame::new(&core.device,
                               &mut core.allocator,
                               &blit_program,
                               &clear_program,
                               &reprojection_program,
                               &stencil_program,
                               quad_vertex_positions_buffer_id,
                               quad_vertex_indices_buffer_id,
                               window_size);

        Renderer {
            core,
            level_impl,

            blit_program,
            clear_program,
            tile_copy_program,
            d3d9_programs,

            frame,

            stencil_program,
            reprojection_program,

            current_cpu_build_time: None,
            pending_timers: VecDeque::new(),
            debug_ui_presenter,
        }
    }

    pub fn begin_scene(&mut self) {
        self.core.framebuffer_flags = FramebufferFlags::empty();

        self.core.device.begin_commands();
        self.core.current_timer = Some(PendingTimer::new());
        self.core.stats = RenderStats::default();

        self.core.alpha_tile_count = 0;
    }

    pub fn render_command(&mut self, command: &RenderCommand) {
        debug!("render command: {:?}", command);
        match *command {
            RenderCommand::Start { bounding_quad, path_count, needs_readable_framebuffer } => {
                self.start_rendering(bounding_quad, path_count, needs_readable_framebuffer);
            }
            RenderCommand::AllocateTexturePage { page_id, ref descriptor } => {
                self.allocate_pattern_texture_page(page_id, descriptor)
            }
            RenderCommand::UploadTexelData { ref texels, location } => {
                self.upload_texel_data(texels, location)
            }
            RenderCommand::DeclareRenderTarget { id, location } => {
                self.declare_render_target(id, location)
            }
            RenderCommand::UploadTextureMetadata(ref metadata) => {
                self.upload_texture_metadata(metadata)
            }
            RenderCommand::AddFillsD3D9(ref fills) => self.add_fills_d3d9(fills),
            RenderCommand::FlushFillsD3D9 => {
                self.draw_buffered_fills_d3d9();
            }
            RenderCommand::UploadSceneD3D11 { ref draw_segments, ref clip_segments } => {
                self.level_impl.require_d3d11().upload_scene_d3d11(&mut self.core,
                                                                   draw_segments,
                                                                   clip_segments)
            }
            RenderCommand::BeginTileDrawing => {}
            RenderCommand::PushRenderTarget(render_target_id) => {
                self.push_render_target(render_target_id)
            }
            RenderCommand::PopRenderTarget => self.pop_render_target(),
            RenderCommand::PrepareClipTilesD3D11(ref batch) => {
                self.level_impl.require_d3d11().prepare_tiles_d3d11(&mut self.core, batch)
            }
            RenderCommand::DrawTilesD3D11(ref batch) => {
                self.level_impl.require_d3d11().prepare_and_draw_tiles_d3d11(&mut self.core, batch)
            }
            RenderCommand::Finish { cpu_build_time } => {
                self.core.stats.cpu_build_time = cpu_build_time;
            }
        }
    }

    pub fn end_scene(&mut self) {
        self.clear_dest_framebuffer_if_necessary();
        self.blit_intermediate_dest_framebuffer_if_necessary();

        self.core.device.end_commands();
        self.core.stats.gpu_bytes_allocated = self.core.allocator.bytes_allocated();

        match self.level_impl {
            RendererLevelImpl::D3D11(ref mut d3d11_renderer) => {
                d3d11_renderer.end_frame(&mut self.core)
            }
        }

        self.core.allocator.dump();

        if let Some(timer) = self.core.current_timer.take() {
            self.pending_timers.push_back(timer);
        }
        self.current_cpu_build_time = None;
    }

    fn start_rendering(&mut self,
                       bounding_quad: BoundingQuad,
                       path_count: usize,
                       needs_readable_framebuffer: bool) {
        match (&self.core.dest_framebuffer, self.core.options.level) {
            (&DestFramebuffer::Other(_), _) => {
                self.core   
                    .renderer_flags
                    .remove(RendererFlags::INTERMEDIATE_DEST_FRAMEBUFFER_NEEDED);
            }
            (&DestFramebuffer::Default { .. }, RendererLevel::D3D11) => {
                self.core
                    .renderer_flags
                    .insert(RendererFlags::INTERMEDIATE_DEST_FRAMEBUFFER_NEEDED);
            }
            _ => {
                self.core
                    .renderer_flags
                    .set(RendererFlags::INTERMEDIATE_DEST_FRAMEBUFFER_NEEDED,
                         needs_readable_framebuffer);
            }
        }

        if self.core.renderer_flags.contains(RendererFlags::USE_DEPTH) {
            self.draw_stencil(&bounding_quad);
        }

        self.core.stats.path_count = path_count;

        self.core.render_targets.clear();
    }

    pub fn draw_debug_ui(&self) {
        self.debug_ui_presenter.draw(&self.core.device);
    }

    pub fn shift_rendering_time(&mut self) -> Option<RenderTime> {
        if let Some(mut pending_timer) = self.pending_timers.pop_front() {
            for old_query in pending_timer.poll(&self.core.device) {
                self.core.timer_query_cache.free(old_query);
            }
            if let Some(render_time) = pending_timer.total_time() {
                return Some(render_time);
            }
            self.pending_timers.push_front(pending_timer);
        }
        None
    }

    #[inline]
    pub fn device(&self) -> &D {
        &self.core.device
    }

    #[inline]
    pub fn device_mut(&mut self) -> &mut D {
        &mut self.core.device
    }

    #[inline]
    pub fn dest_framebuffer(&self) -> &DestFramebuffer<D> {
        &self.core.dest_framebuffer
    }

    #[inline]
    pub fn debug_ui_presenter_mut(&mut self) -> (&mut D, &mut DebugUIPresenter<D>) {
        (&mut self.core.device, &mut self.debug_ui_presenter)
    }

    #[inline]
    pub fn replace_dest_framebuffer(&mut self, new_dest_framebuffer: DestFramebuffer<D>)
                                    -> DestFramebuffer<D> {
        mem::replace(&mut self.core.dest_framebuffer, new_dest_framebuffer)
    }

    #[inline]
    pub fn level(&self) -> RendererLevel {
        self.core.options.level
    }

    #[inline]
    pub fn set_options(&mut self, new_options: RendererOptions) {
        self.core.options = new_options
    }

    #[inline]
    pub fn set_main_framebuffer_size(&mut self, new_framebuffer_size: Vector2I) {
        self.debug_ui_presenter.ui_presenter.set_framebuffer_size(new_framebuffer_size);
    }

    #[inline]
    pub fn disable_depth(&mut self) {
        self.core.renderer_flags.remove(RendererFlags::USE_DEPTH);
    }

    #[inline]
    pub fn enable_depth(&mut self) {
        self.core.renderer_flags.insert(RendererFlags::USE_DEPTH);
    }

    #[inline]
    pub fn quad_vertex_positions_buffer(&self) -> &D::Buffer {
        self.core.allocator.get_buffer(self.core.quad_vertex_positions_buffer_id)
    }

    #[inline]
    pub fn quad_vertex_indices_buffer(&self) -> &D::Buffer {
        self.core.allocator.get_buffer(self.core.quad_vertex_indices_buffer_id)
    }

    fn allocate_pattern_texture_page(&mut self,
                                     page_id: TexturePageId,
                                     descriptor: &TexturePageDescriptor) {
        // Fill in IDs up to the requested page ID.
        let page_index = page_id.0 as usize;
        while self.core.pattern_texture_pages.len() < page_index + 1 {
            self.core.pattern_texture_pages.push(None);
        }

        // Clear out any existing texture.
        if let Some(old_texture_page) = self.core.pattern_texture_pages[page_index].take() {
            self.core.allocator.free_framebuffer(old_texture_page.framebuffer_id);
        }

        // Allocate texture.
        let texture_size = descriptor.size;
        let framebuffer_id = self.core.allocator.allocate_framebuffer(&self.core.device,
                                                                      texture_size,
                                                                      TextureFormat::RGBA8,
                                                                      FramebufferTag::PatternPage);
        self.core.pattern_texture_pages[page_index] = Some(PatternTexturePage {
            framebuffer_id,
            must_preserve_contents: false,
        });
    }

    fn upload_texel_data(&mut self, texels: &[ColorU], location: TextureLocation) {
        let texture_page = self.core
                               .pattern_texture_pages[location.page.0 as usize]
                               .as_mut()
                               .expect("Texture page not allocated yet!");
        let framebuffer_id = texture_page.framebuffer_id;
        let framebuffer = self.core.allocator.get_framebuffer(framebuffer_id);
        let texture = self.core.device.framebuffer_texture(framebuffer);
        let texels = color::color_slice_to_u8_slice(texels);
        self.core.device.upload_to_texture(texture, location.rect, TextureDataRef::U8(texels));
        texture_page.must_preserve_contents = true;
    }

    fn declare_render_target(&mut self,
                             render_target_id: RenderTargetId,
                             location: TextureLocation) {
        while self.core.render_targets.len() < render_target_id.render_target as usize + 1 {
            self.core.render_targets.push(RenderTargetInfo {
                location: TextureLocation { page: TexturePageId(!0), rect: RectI::default() },
            });
        }
        let mut render_target =
            &mut self.core.render_targets[render_target_id.render_target as usize];
        debug_assert_eq!(render_target.location.page, TexturePageId(!0));
        render_target.location = location;
    }

    fn upload_texture_metadata(&mut self, metadata: &[TextureMetadataEntry]) {
        let padded_texel_size =
            (util::alignup_i32(metadata.len() as i32, TEXTURE_METADATA_ENTRIES_PER_ROW) *
             TEXTURE_METADATA_TEXTURE_WIDTH * 4) as usize;
        let mut texels = Vec::with_capacity(padded_texel_size);
        for entry in metadata {
            let base_color = entry.base_color.to_f32();
            texels.extend_from_slice(&[
                f16::from_f32(entry.color_0_transform.m11()),
                f16::from_f32(entry.color_0_transform.m21()),
                f16::from_f32(entry.color_0_transform.m12()),
                f16::from_f32(entry.color_0_transform.m22()),
                f16::from_f32(entry.color_0_transform.m13()),
                f16::from_f32(entry.color_0_transform.m23()),
                f16::default(),
                f16::default(),
                f16::from_f32(base_color.r()),
                f16::from_f32(base_color.g()),
                f16::from_f32(base_color.b()),
                f16::from_f32(base_color.a()),
                f16::default(),
                f16::default(),
                f16::default(),
                f16::default(),
            ]);
        }
        while texels.len() < padded_texel_size {
            texels.push(f16::default())
        }

        let texture_id = self.core.texture_metadata_texture_id;
        let texture = self.core.allocator.get_texture(texture_id);
        let width = TEXTURE_METADATA_TEXTURE_WIDTH;
        let height = texels.len() as i32 / (4 * TEXTURE_METADATA_TEXTURE_WIDTH);
        let rect = RectI::new(Vector2I::zero(), Vector2I::new(width, height));
        self.core.device.upload_to_texture(texture, rect, TextureDataRef::F16(&texels));
    }

    fn allocate_tiles_d3d9(&mut self, tile_count: u32) -> BufferID {
        /*
        let device = &self.device;
        let tile_program = &self.tile_program;
        let tile_copy_program = &self.tile_copy_program;
        let quad_vertex_positions_buffer = &self.quad_vertex_positions_buffer;
        let quad_vertex_indices_buffer = &self.quad_vertex_indices_buffer;
        self.back_frame.storage_allocators.tile_vertex.allocate(tile_count as u64, |size| {
            TileVertexStorage::new(size,
                                   device,
                                   tile_program,
                                   tile_copy_program,
                                   quad_vertex_positions_buffer,
                                   quad_vertex_indices_buffer)
        })
        */
        unimplemented!()
    }

    fn upload_tiles_d3d9(&mut self, buffer_id: BufferID, tiles: &[TileObjectPrimitive]) {
        /*
        let vertex_buffer = &self.back_frame
                                 .storage_allocators
                                 .tile_vertex
                                 .get(storage_id)
                                 .vertex_buffer;
        self.device.upload_to_buffer(vertex_buffer, 0, tiles, BufferTarget::Vertex);

        self.ensure_index_buffer(tiles.len());
        */
        unimplemented!()
    }


    fn ensure_index_buffer(&mut self, mut length: usize) {
        length = length.next_power_of_two();
        if self.frame.quads_vertex_indices_length >= length {
            return;
        }

        // TODO(pcwalton): Generate these with SIMD.
        let mut indices: Vec<u32> = Vec::with_capacity(length * 6);
        for index in 0..(length as u32) {
            indices.extend_from_slice(&[
                index * 4 + 0, index * 4 + 1, index * 4 + 2,
                index * 4 + 1, index * 4 + 3, index * 4 + 2,
            ]);
        }

        if let Some(quads_vertex_indices_buffer_id) = self.frame
                                                          .quads_vertex_indices_buffer_id
                                                          .take() {
            self.core.allocator.free_buffer(quads_vertex_indices_buffer_id);
        }
        let quads_vertex_indices_buffer_id =
            self.core.allocator.allocate_buffer::<u32>(&self.core.device,
                                                       indices.len() as u64,
                                                       BufferTag::QuadsVertexIndices);
        let quads_vertex_indices_buffer =
            self.core.allocator.get_buffer(quads_vertex_indices_buffer_id);
        self.core.device.upload_to_buffer(quads_vertex_indices_buffer,
                                          0,
                                          &indices,
                                          BufferTarget::Index);
        self.frame.quads_vertex_indices_buffer_id = Some(quads_vertex_indices_buffer_id);
        self.frame.quads_vertex_indices_length = length;
    }

    fn add_fills_d3d9(&mut self, fill_batch: &[Fill]) {
        if fill_batch.is_empty() {
            return;
        }

        self.core.stats.fill_count += fill_batch.len();

        let preserve_alpha_mask_contents = self.core.alpha_tile_count > 0;

        self.frame.pending_fills.reserve(fill_batch.len());
        for fill in fill_batch {
            self.core.alpha_tile_count = self.core.alpha_tile_count.max(fill.link + 1);
            self.frame.pending_fills.push(*fill);
        }

        self.core.reallocate_alpha_tile_pages_if_necessary(preserve_alpha_mask_contents);

        if self.frame.buffered_fills.len() + self.frame.pending_fills.len() > MAX_FILLS_PER_BATCH {
            self.draw_buffered_fills_d3d9();
        }

        self.frame.buffered_fills.extend(self.frame.pending_fills.drain(..));
    }

    fn draw_buffered_fills_d3d9(&mut self) {
        if self.frame.buffered_fills.is_empty() {
            return;
        }

        let fill_storage_info = self.upload_buffered_fills_d3d9();
        self.draw_fills_d3d9(fill_storage_info.fill_buffer_id, fill_storage_info.fill_count);
    }

    fn upload_buffered_fills_d3d9(&mut self) -> FillBufferInfoD3D9 {
        /*
        let buffered_fills = &mut self.back_frame.buffered_fills;
        debug_assert!(!buffered_fills.is_empty());

        let storage_id = {
            let device = &self.device;
            let fill_program = &self.fill_program;
            let quad_vertex_positions_buffer = &self.quad_vertex_positions_buffer;
            let quad_vertex_indices_buffer = &self.quad_vertex_indices_buffer;
            let renderer_level = self.options.level;

            self.back_frame.storage_allocators.fill_vertex.allocate(MAX_FILLS_PER_BATCH as u64,
                                                                    |size| {
                FillVertexStorage::new(size,
                                       device,
                                       fill_program,
                                       quad_vertex_positions_buffer,
                                       quad_vertex_indices_buffer,
                                       renderer_level)
            })
        };
        let fill_vertex_storage = self.back_frame.storage_allocators.fill_vertex.get(storage_id);

        debug_assert!(buffered_fills.len() <= u32::MAX as usize);
        self.device.upload_to_buffer(&fill_vertex_storage.vertex_buffer,
                                     0,
                                     &buffered_fills,
                                     BufferTarget::Vertex);

        let fill_count = buffered_fills.len() as u32;
        buffered_fills.clear();

        FillRasterStorageInfo { fill_storage_id: storage_id, fill_count }
        */
        unreachable!()
    }

    fn draw_fills_d3d9(&mut self, fill_buffer_id: BufferID, fill_count: u32) {
        /*
        let fill_raster_program = match self.fill_program {
            FillProgram::Raster(ref fill_raster_program) => fill_raster_program,
            _ => unreachable!(),
        };
        let mask_viewport = self.mask_viewport();
        let fill_vertex_storage = self.back_frame
                                      .storage_allocators
                                      .fill_vertex
                                      .get(fill_storage_id);
        let fill_vertex_array =
            fill_vertex_storage.vertex_array.as_ref().expect("Where's the vertex array?");

        let mut clear_color = None;
        if !self.back_frame
                .framebuffer_flags
                .contains(FramebufferFlags::MASK_FRAMEBUFFER_IS_DIRTY) {
            clear_color = Some(ColorF::default());
        };

        let timer_query = self.timer_query_cache.alloc(&self.device);
        self.device.begin_timer_query(&timer_query);

        self.device.draw_elements_instanced(6, fill_count, &RenderState {
            target: &RenderTarget::Framebuffer(&self.back_frame
                                                    .mask_storage
                                                    .as_ref()
                                                    .expect("Where's the mask storage?")
                                                    .framebuffer),
            program: &fill_raster_program.program,
            vertex_array: &fill_vertex_array.vertex_array,
            primitive: Primitive::Triangles,
            textures: &[(&fill_raster_program.area_lut_texture, &self.area_lut_texture)],
            uniforms: &[
                (&fill_raster_program.framebuffer_size_uniform,
                 UniformData::Vec2(mask_viewport.size().to_f32().0)),
                (&fill_raster_program.tile_size_uniform,
                 UniformData::Vec2(F32x2::new(TILE_WIDTH as f32, TILE_HEIGHT as f32))),
            ],
            images: &[],
            storage_buffers: &[],
            viewport: mask_viewport,
            options: RenderOptions {
                blend: Some(BlendState {
                    src_rgb_factor: BlendFactor::One,
                    src_alpha_factor: BlendFactor::One,
                    dest_rgb_factor: BlendFactor::One,
                    dest_alpha_factor: BlendFactor::One,
                    ..BlendState::default()
                }),
                clear_ops: ClearOps { color: clear_color, ..ClearOps::default() },
                ..RenderOptions::default()
            },
        });

        self.device.end_timer_query(&timer_query);
        self.current_timer.as_mut().unwrap().raster_times.push(TimerFuture::new(timer_query));
        self.stats.drawcall_count += 1;

        self.back_frame.framebuffer_flags.insert(FramebufferFlags::MASK_FRAMEBUFFER_IS_DIRTY);
        */
        unreachable!()
    }

    /*
    fn clip_tiles_d3d9(&mut self, clip_storage_id: StorageID, max_clipped_tile_count: u32) {
        let mask_framebuffer = &self.back_frame
                                    .mask_storage
                                    .as_ref()
                                    .expect("Where's the mask storage?")
                                    .framebuffer;
        let mask_texture = self.device.framebuffer_texture(mask_framebuffer);
        let mask_texture_size = self.device.texture_size(&mask_texture);

        // Allocate temp mask framebuffer if necessary.
        match self.back_frame.mask_temp_framebuffer {
            Some(ref mask_temp_framebuffer) if
                self.device.texture_size(
                    self.device.framebuffer_texture(
                        mask_temp_framebuffer)).y() >= mask_texture_size.y() => {}
            _ => {
                let mask_texture_format = self.mask_texture_format();
                let mask_temp_texture = self.device.create_texture(mask_texture_format,
                                                                   mask_texture_size);
                self.back_frame.mask_temp_framebuffer =
                    Some(self.device.create_framebuffer(mask_temp_texture));
            }
        }
        let mask_temp_framebuffer = self.back_frame.mask_temp_framebuffer.as_ref().unwrap();

        let clip_vertex_storage = self.allocator.get(clip_storage_id);

        let timer_query = self.timer_query_cache.alloc(&self.device);
        self.device.begin_timer_query(&timer_query);

        // Copy out tiles.
        //
        // TODO(pcwalton): Don't do this on GL4.
        self.device.draw_elements_instanced(6, max_clipped_tile_count * 2, &RenderState {
            target: &RenderTarget::Framebuffer(mask_temp_framebuffer),
            program: &self.tile_clip_copy_program.program,
            vertex_array: &clip_vertex_storage.tile_clip_copy_vertex_array.vertex_array,
            primitive: Primitive::Triangles,
            textures: &[
                (&self.tile_clip_copy_program.src_texture,
                 self.device.framebuffer_texture(mask_framebuffer)),
            ],
            images: &[],
            uniforms: &[
                (&self.tile_clip_copy_program.framebuffer_size_uniform,
                 UniformData::Vec2(mask_texture_size.to_f32().0)),
            ],
            storage_buffers: &[],
            viewport: RectI::new(Vector2I::zero(), mask_texture_size),
            options: RenderOptions::default(),
        });

        // Combine clip tiles.
        self.device.draw_elements_instanced(6, max_clipped_tile_count, &RenderState {
            target: &RenderTarget::Framebuffer(mask_framebuffer),
            program: &self.tile_clip_combine_program.program,
            vertex_array: &clip_vertex_storage.tile_clip_combine_vertex_array.vertex_array,
            primitive: Primitive::Triangles,
            textures: &[
                (&self.tile_clip_combine_program.src_texture,
                 self.device.framebuffer_texture(&mask_temp_framebuffer)),
            ],
            images: &[],
            uniforms: &[
                (&self.tile_clip_combine_program.framebuffer_size_uniform,
                 UniformData::Vec2(mask_texture_size.to_f32().0)),
            ],
            storage_buffers: &[],
            viewport: RectI::new(Vector2I::zero(), mask_texture_size),
            options: RenderOptions::default(),
        });

        self.device.end_timer_query(&timer_query);
        self.current_timer.as_mut().unwrap().raster_times.push(TimerFuture::new(timer_query));
        self.stats.drawcall_count += 2;
    }
    */

    fn tile_transform(&self) -> Transform4F {
        let draw_viewport = self.core.draw_viewport().size().to_f32();
        let scale = Vector4F::new(2.0 / draw_viewport.x(), -2.0 / draw_viewport.y(), 1.0, 1.0);
        Transform4F::from_scale(scale).translate(Vector4F::new(-1.0, 1.0, 0.0, 1.0))
    }

    fn upload_z_buffer_d3d9(&mut self,
                            z_buffer_texture_id: TextureID,
                            z_buffer_map: &DenseTileMap<i32>) {
        let z_buffer_texture = self.core.allocator.get_texture(z_buffer_texture_id);
        debug_assert_eq!(z_buffer_map.rect.origin(), Vector2I::default());
        debug_assert_eq!(z_buffer_map.rect.size(),
                         self.core.device.texture_size(z_buffer_texture));
        let z_data: &[u8] = z_buffer_map.data.as_byte_slice();
        self.core.device.upload_to_texture(z_buffer_texture,
                                           z_buffer_map.rect,
                                           TextureDataRef::U8(&z_data));
    }

    /*
    fn allocate_clip_storage(&mut self, max_clipped_tile_count: u32) -> BufferID {
        let device = &self.device;
        let tile_clip_combine_program = &self.tile_clip_combine_program;
        let tile_clip_copy_program = &self.tile_clip_copy_program;
        let quad_vertex_positions_buffer = &self.quad_vertex_positions_buffer;
        let quad_vertex_indices_buffer = &self.quad_vertex_indices_buffer;
        self.back_frame.storage_allocators.clip_vertex.allocate(max_clipped_tile_count as u64,
                                                                |size| {
            ClipVertexStorage::new(size,
                                   device,
                                   tile_clip_combine_program,
                                   tile_clip_copy_program,
                                   quad_vertex_positions_buffer,
                                   quad_vertex_indices_buffer)
        })
    }

    // Uploads clip tiles from CPU to GPU.
    fn upload_clip_tiles(&mut self, clip_vertex_storage_id: StorageID, clips: &[Clip]) {
        let clip_vertex_storage = self.back_frame
                                      .storage_allocators
                                      .clip_vertex
                                      .get(clip_vertex_storage_id);
        self.device.upload_to_buffer(&clip_vertex_storage.vertex_buffer,
                                     0,
                                     clips,
                                     BufferTarget::Vertex);
    }
    */

    /*
    fn draw_tiles_d3d9(&mut self,
                       tile_count: u32,
                       storage_id: StorageID,
                       color_texture_0: Option<TileBatchTexture>,
                       blend_mode: BlendMode,
                       filter: Filter,
                       z_buffer_storage_id: StorageID) {
        // TODO(pcwalton): Disable blend for solid tiles.

        let needs_readable_framebuffer = blend_mode.needs_readable_framebuffer();
        if needs_readable_framebuffer {
            self.copy_alpha_tiles_to_dest_blend_texture(tile_count, storage_id);
        }

        let clear_color = self.clear_color_for_draw_operation();
        let draw_viewport = self.draw_viewport();

        let timer_query = self.timer_query_cache.alloc(&self.device);
        self.device.begin_timer_query(&timer_query);

        let tile_raster_program = match self.tile_program {
            TileProgram::Raster(ref tile_raster_program) => tile_raster_program,
            TileProgram::Compute(_) => unreachable!(),
        };

        let (mut textures, mut uniforms) = (vec![], vec![]);

        self.set_uniforms_for_drawing_tiles(&tile_raster_program.common,
                                            &mut textures,
                                            &mut uniforms,
                                            color_texture_0,
                                            blend_mode,
                                            filter,
                                            z_buffer_storage_id);

        uniforms.push((&tile_raster_program.transform_uniform,
                       UniformData::Mat4(self.tile_transform().to_columns())));

        if needs_readable_framebuffer {
            textures.push((&tile_raster_program.dest_texture,
                           self.device
                               .framebuffer_texture(&self.back_frame.dest_blend_framebuffer)));
        }

        let vertex_array = &self.back_frame
                                .storage_allocators
                                .tile_vertex
                                .get(storage_id)
                                .tile_vertex_array
                                .as_ref()
                                .expect("No tile vertex array present!")
                                .vertex_array;

        self.device.draw_elements_instanced(6, tile_count, &RenderState {
            target: &self.draw_render_target(),
            program: &tile_raster_program.common.program,
            vertex_array,
            primitive: Primitive::Triangles,
            textures: &textures,
            images: &[],
            storage_buffers: &[],
            uniforms: &uniforms,
            viewport: draw_viewport,
            options: RenderOptions {
                blend: blend_mode.to_blend_state(),
                stencil: self.stencil_state(),
                clear_ops: ClearOps { color: clear_color, ..ClearOps::default() },
                ..RenderOptions::default()
            },
        });

        self.device.end_timer_query(&timer_query);
        self.current_timer.as_mut().unwrap().raster_times.push(TimerFuture::new(timer_query));
        self.stats.drawcall_count += 1;

        self.preserve_draw_framebuffer();
    }
    */

    fn copy_alpha_tiles_to_dest_blend_texture(&mut self, tile_count: u32, buffer_id: BufferID) {
        /*
        let draw_viewport = self.draw_viewport();

        let mut textures = vec![];
        let mut uniforms = vec![
            (&self.tile_copy_program.transform_uniform,
             UniformData::Mat4(self.tile_transform().to_columns())),
            (&self.tile_copy_program.tile_size_uniform,
             UniformData::Vec2(F32x2::new(TILE_WIDTH as f32, TILE_HEIGHT as f32))),
        ];

        let draw_framebuffer = match self.draw_render_target() {
            RenderTarget::Framebuffer(framebuffer) => framebuffer,
            RenderTarget::Default => panic!("Can't copy alpha tiles from default framebuffer!"),
        };
        let draw_texture = self.device.framebuffer_texture(&draw_framebuffer);

        textures.push((&self.tile_copy_program.src_texture, draw_texture));
        uniforms.push((&self.tile_copy_program.framebuffer_size_uniform,
                       UniformData::Vec2(draw_viewport.size().to_f32().0)));

        let vertex_array = &self.back_frame
                                .storage_allocators
                                .tile_vertex
                                .get(storage_id)
                                .tile_copy_vertex_array
                                .vertex_array;

        self.device.draw_elements(tile_count * 6, &RenderState {
            target: &RenderTarget::Framebuffer(&self.back_frame.dest_blend_framebuffer),
            program: &self.tile_copy_program.program,
            vertex_array,
            primitive: Primitive::Triangles,
            textures: &textures,
            images: &[],
            storage_buffers: &[],
            uniforms: &uniforms,
            viewport: draw_viewport,
            options: RenderOptions {
                clear_ops: ClearOps {
                    color: Some(ColorF::new(1.0, 0.0, 0.0, 1.0)),
                    ..ClearOps::default()
                },
                ..RenderOptions::default()
            },
        });

        self.stats.drawcall_count += 1;
        */
        unimplemented!()
    }

    fn draw_stencil(&mut self, quad_positions: &[Vector4F]) {
        self.core.device.allocate_buffer(&self.frame.stencil_vertex_array.vertex_buffer,
                                         BufferData::Memory(quad_positions),
                                         BufferTarget::Vertex);

        // Create indices for a triangle fan. (This is OK because the clipped quad should always be
        // convex.)
        let mut indices: Vec<u32> = vec![];
        for index in 1..(quad_positions.len() as u32 - 1) {
            indices.extend_from_slice(&[0, index as u32, index + 1]);
        }
        self.core.device.allocate_buffer(&self.frame.stencil_vertex_array.index_buffer,
                                    BufferData::Memory(&indices),
                                    BufferTarget::Index);

        self.core.device.draw_elements(indices.len() as u32, &RenderState {
            target: &self.core.draw_render_target(),
            program: &self.stencil_program.program,
            vertex_array: &self.frame.stencil_vertex_array.vertex_array,
            primitive: Primitive::Triangles,
            textures: &[],
            images: &[],
            storage_buffers: &[],
            uniforms: &[],
            viewport: self.core.draw_viewport(),
            options: RenderOptions {
                // FIXME(pcwalton): Should we really write to the depth buffer?
                depth: Some(DepthState { func: DepthFunc::Less, write: true }),
                stencil: Some(StencilState {
                    func: StencilFunc::Always,
                    reference: 1,
                    mask: 1,
                    write: true,
                }),
                color_mask: false,
                clear_ops: ClearOps { stencil: Some(0), ..ClearOps::default() },
                ..RenderOptions::default()
            },
        });

        self.core.stats.drawcall_count += 1;
    }

    pub fn reproject_texture(&mut self,
                             texture: &D::Texture,
                             old_transform: &Transform4F,
                             new_transform: &Transform4F) {
        let clear_color = self.core.clear_color_for_draw_operation();

        self.core.device.draw_elements(6, &RenderState {
            target: &self.core.draw_render_target(),
            program: &self.reprojection_program.program,
            vertex_array: &self.frame.reprojection_vertex_array.vertex_array,
            primitive: Primitive::Triangles,
            textures: &[(&self.reprojection_program.texture, texture)],
            images: &[],
            storage_buffers: &[],
            uniforms: &[
                (&self.reprojection_program.old_transform_uniform,
                 UniformData::from_transform_3d(old_transform)),
                (&self.reprojection_program.new_transform_uniform,
                 UniformData::from_transform_3d(new_transform)),
            ],
            viewport: self.core.draw_viewport(),
            options: RenderOptions {
                blend: BlendMode::SrcOver.to_blend_state(),
                depth: Some(DepthState { func: DepthFunc::Less, write: false, }),
                clear_ops: ClearOps { color: clear_color, ..ClearOps::default() },
                ..RenderOptions::default()
            },
        });

        self.core.stats.drawcall_count += 1;

        self.core.preserve_draw_framebuffer();
    }

    fn push_render_target(&mut self, render_target_id: RenderTargetId) {
        self.core.render_target_stack.push(render_target_id);
    }

    fn pop_render_target(&mut self) {
        self.core.render_target_stack.pop().expect("Render target stack underflow!");
    }

    fn clear_dest_framebuffer_if_necessary(&mut self) {
        let background_color = match self.core.options.background_color {
            None => return,
            Some(background_color) => background_color,
        };

        if self.core.framebuffer_flags.contains(FramebufferFlags::DEST_FRAMEBUFFER_IS_DIRTY) {
            return;
        }

        let main_viewport = self.core.main_viewport();
        let uniforms = [
            (&self.clear_program.rect_uniform, UniformData::Vec4(main_viewport.to_f32().0)),
            (&self.clear_program.framebuffer_size_uniform,
             UniformData::Vec2(main_viewport.size().to_f32().0)),
            (&self.clear_program.color_uniform, UniformData::Vec4(background_color.0)),
        ];

        self.core.device.draw_elements(6, &RenderState {
            target: &RenderTarget::Default,
            program: &self.clear_program.program,
            vertex_array: &self.frame.clear_vertex_array.vertex_array,
            primitive: Primitive::Triangles,
            textures: &[],
            images: &[],
            storage_buffers: &[],
            uniforms: &uniforms[..],
            viewport: main_viewport,
            options: RenderOptions::default(),
        });

        self.core.stats.drawcall_count += 1;
    }

    fn blit_intermediate_dest_framebuffer_if_necessary(&mut self) {
        if !self.core
                .renderer_flags
                .contains(RendererFlags::INTERMEDIATE_DEST_FRAMEBUFFER_NEEDED) {
            return;
        }

        let main_viewport = self.core.main_viewport();

        let intermediate_dest_framebuffer =
            self.core.allocator.get_framebuffer(self.core.intermediate_dest_framebuffer_id);

        let textures = [
            (&self.blit_program.src_texture,
             self.core.device.framebuffer_texture(intermediate_dest_framebuffer))
        ];

        self.core.device.draw_elements(6, &RenderState {
            target: &RenderTarget::Default,
            program: &self.blit_program.program,
            vertex_array: &self.frame.blit_vertex_array.vertex_array,
            primitive: Primitive::Triangles,
            textures: &textures[..],
            images: &[],
            storage_buffers: &[],
            uniforms: &[
                (&self.blit_program.framebuffer_size_uniform,
                 UniformData::Vec2(main_viewport.size().to_f32().0)),
                (&self.blit_program.dest_rect_uniform,
                 UniformData::Vec4(RectF::new(Vector2F::zero(), main_viewport.size().to_f32()).0)),
            ],
            viewport: main_viewport,
            options: RenderOptions {
                clear_ops: ClearOps {
                    color: Some(ColorF::new(0.0, 0.0, 0.0, 1.0)),
                    ..ClearOps::default()
                },
                ..RenderOptions::default()
            },
        });

        self.core.stats.drawcall_count += 1;
    }

    fn stencil_state(&self) -> Option<StencilState> {
        if !self.core.renderer_flags.contains(RendererFlags::USE_DEPTH) {
            return None;
        }

        Some(StencilState {
            func: StencilFunc::Equal,
            reference: 1,
            mask: 1,
            write: false,
        })
    }

    fn mask_viewport(&self) -> RectI {
        let page_count = match self.core.mask_storage {
            Some(ref mask_storage) => mask_storage.allocated_page_count as i32,
            None => 0,
        };
        let height = MASK_FRAMEBUFFER_HEIGHT * page_count;
        RectI::new(Vector2I::default(), vec2i(MASK_FRAMEBUFFER_WIDTH, height))
    }

    #[inline]
    pub fn draw_viewport(&self) -> RectI {
        self.core.draw_viewport()
    }

    #[inline]
    pub fn draw_render_target(&self) -> RenderTarget<D> {
        self.core.draw_render_target()
    }

    #[inline]
    pub fn render_stats(&self) -> &RenderStats {
        &self.core.stats
    }
}

impl<D> RendererCore<D> where D: Device {
    fn mask_texture_format(&self) -> TextureFormat {
        match self.options.level {
            RendererLevel::D3D9 => TextureFormat::RGBA16F,
            RendererLevel::D3D11 => TextureFormat::RGBA8,
        }
    }

    pub(crate) fn reallocate_alpha_tile_pages_if_necessary(&mut self, copy_existing: bool) {
        let alpha_tile_pages_needed = ((self.alpha_tile_count + 0xffff) >> 16) as u32;
        if let Some(ref mask_storage) = self.mask_storage {
            if alpha_tile_pages_needed <= mask_storage.allocated_page_count {
                return;
            }
        }

        //println!("*** reallocating alpha tile pages");

        let new_size = vec2i(MASK_FRAMEBUFFER_WIDTH,
                             MASK_FRAMEBUFFER_HEIGHT * alpha_tile_pages_needed as i32);
        let format = self.mask_texture_format();
        let mask_framebuffer_id =
            self.allocator.allocate_framebuffer(&self.device,
                                                new_size,
                                                format,
                                                FramebufferTag::TileAlphaMask);
        let mask_framebuffer = self.allocator.get_framebuffer(mask_framebuffer_id);
        let mask_texture = self.device.framebuffer_texture(&mask_framebuffer);
        let old_mask_storage = self.mask_storage.take();
        self.mask_storage = Some(MaskStorage {
            framebuffer_id: mask_framebuffer_id,
            allocated_page_count: alpha_tile_pages_needed,
        });

        // Copy over existing content if needed.
        let old_mask_framebuffer_id = match old_mask_storage {
            Some(old_storage) if copy_existing => old_storage.framebuffer_id,
            Some(_) | None => return,
        };
        let old_mask_framebuffer = self.allocator.get_framebuffer(old_mask_framebuffer_id);
        let old_mask_texture = self.device.framebuffer_texture(old_mask_framebuffer);
        let old_size = self.device.texture_size(old_mask_texture);

        let timer_query = self.timer_query_cache.alloc(&self.device);
        self.device.begin_timer_query(&timer_query);

        self.device.draw_elements(6, &RenderState {
            target: &RenderTarget::Framebuffer(mask_framebuffer),
            program: &self.programs.blit_program.program,
            vertex_array: &self.vertex_arrays.blit_vertex_array.vertex_array,
            primitive: Primitive::Triangles,
            textures: &[(&self.programs.blit_program.src_texture, old_mask_texture)],
            images: &[],
            storage_buffers: &[],
            uniforms: &[
                (&self.programs.blit_program.framebuffer_size_uniform,
                 UniformData::Vec2(new_size.to_f32().0)),
                (&self.programs.blit_program.dest_rect_uniform,
                 UniformData::Vec4(RectF::new(Vector2F::zero(), old_size.to_f32()).0)),
            ],
            viewport: RectI::new(Vector2I::default(), new_size),
            options: RenderOptions {
                clear_ops: ClearOps {
                    color: Some(ColorF::new(0.0, 0.0, 0.0, 1.0)),
                    ..ClearOps::default()
                },
                ..RenderOptions::default()
            },
        });

        self.device.end_timer_query(&timer_query);
        self.current_timer.as_mut().unwrap().other_times.push(TimerFuture::new(timer_query));
        self.stats.drawcall_count += 1;
    }

    pub(crate) fn set_uniforms_for_drawing_tiles<'a>(
            &'a self,
            tile_program: &'a TileProgramCommon<D>,
            textures: &mut Vec<TextureBinding<'a, D::TextureParameter, D::Texture>>,
            uniforms: &mut Vec<UniformBinding<'a, D::Uniform>>,
            color_texture_0: Option<TileBatchTexture>,
            blend_mode: BlendMode,
            filter: Filter,
            z_buffer_id: BufferID) {
        let draw_viewport = self.draw_viewport();

        let z_buffer = self.allocator.get_buffer(z_buffer_id);
        // FIXME(pcwalton)
        //let z_buffer_texture = self.device.framebuffer_texture(&z_buffer.framebuffer);

        let texture_metadata_texture =
            self.allocator.get_texture(self.texture_metadata_texture_id);
        textures.push((&tile_program.texture_metadata_texture, texture_metadata_texture));

        //textures.push((&tile_program.z_buffer_texture, z_buffer_texture));

        /*
        uniforms.push((&tile_program.z_buffer_texture_size_uniform,
                       UniformData::IVec2(self.device.texture_size(z_buffer_texture).0)));
        */
        uniforms.push((&tile_program.tile_size_uniform,
                       UniformData::Vec2(F32x2::new(TILE_WIDTH as f32, TILE_HEIGHT as f32))));
        uniforms.push((&tile_program.framebuffer_size_uniform,
                       UniformData::Vec2(draw_viewport.size().to_f32().0)));
        uniforms.push((&tile_program.texture_metadata_size_uniform,
                       UniformData::IVec2(I32x2::new(TEXTURE_METADATA_TEXTURE_WIDTH,
                                                     TEXTURE_METADATA_TEXTURE_HEIGHT))));

        if let Some(ref mask_storage) = self.mask_storage {
            let mask_framebuffer_id = mask_storage.framebuffer_id;
            let mask_framebuffer = self.allocator.get_framebuffer(mask_framebuffer_id);
            let mask_texture = self.device.framebuffer_texture(mask_framebuffer);
            uniforms.push((&tile_program.mask_texture_size_0_uniform,
                           UniformData::Vec2(self.device.texture_size(mask_texture).to_f32().0)));
            textures.push((&tile_program.mask_texture_0, mask_texture));
        }

        // TODO(pcwalton): Refactor.
        let mut ctrl = 0;
        match color_texture_0 {
            Some(color_texture) => {
                let color_texture_page = self.texture_page(color_texture.page);
                let color_texture_size = self.device.texture_size(color_texture_page).to_f32();
                self.device.set_texture_sampling_mode(color_texture_page,
                                                      color_texture.sampling_flags);
                textures.push((&tile_program.color_texture_0, color_texture_page));
                uniforms.push((&tile_program.color_texture_size_0_uniform,
                               UniformData::Vec2(color_texture_size.0)));

                ctrl |= color_texture.composite_op.to_combine_mode() <<
                    COMBINER_CTRL_COLOR_COMBINE_SHIFT;
            }
            None => {
                uniforms.push((&tile_program.color_texture_size_0_uniform,
                               UniformData::Vec2(F32x2::default())));
            }
        }

        ctrl |= blend_mode.to_composite_ctrl() << COMBINER_CTRL_COMPOSITE_SHIFT;

        match filter {
            Filter::None => self.set_uniforms_for_no_filter(tile_program, uniforms),
            Filter::RadialGradient { line, radii, uv_origin } => {
                ctrl |= COMBINER_CTRL_FILTER_RADIAL_GRADIENT << COMBINER_CTRL_COLOR_FILTER_SHIFT;
                self.set_uniforms_for_radial_gradient_filter(tile_program,
                                                             uniforms,
                                                             line,
                                                             radii,
                                                             uv_origin)
            }
            Filter::PatternFilter(PatternFilter::Text {
                fg_color,
                bg_color,
                defringing_kernel,
                gamma_correction,
            }) => {
                ctrl |= COMBINER_CTRL_FILTER_TEXT << COMBINER_CTRL_COLOR_FILTER_SHIFT;
                self.set_uniforms_for_text_filter(tile_program,
                                                  textures,
                                                  uniforms,
                                                  fg_color,
                                                  bg_color,
                                                  defringing_kernel,
                                                  gamma_correction);
            }
            Filter::PatternFilter(PatternFilter::Blur { direction, sigma }) => {
                ctrl |= COMBINER_CTRL_FILTER_BLUR << COMBINER_CTRL_COLOR_FILTER_SHIFT;
                self.set_uniforms_for_blur_filter(tile_program, uniforms, direction, sigma);
            }
        }

        uniforms.push((&tile_program.ctrl_uniform, UniformData::Int(ctrl)));
    }

    fn set_uniforms_for_no_filter<'a>(&'a self,
                                      tile_program: &'a TileProgramCommon<D>,
                                      uniforms: &mut Vec<(&'a D::Uniform, UniformData)>) {
        uniforms.extend_from_slice(&[
            (&tile_program.filter_params_0_uniform, UniformData::Vec4(F32x4::default())),
            (&tile_program.filter_params_1_uniform, UniformData::Vec4(F32x4::default())),
            (&tile_program.filter_params_2_uniform, UniformData::Vec4(F32x4::default())),
        ]);
    }

    fn set_uniforms_for_radial_gradient_filter<'a>(
            &'a self,
            tile_program: &'a TileProgramCommon<D>,
            uniforms: &mut Vec<(&'a D::Uniform, UniformData)>,
            line: LineSegment2F,
            radii: F32x2,
            uv_origin: Vector2F) {
        uniforms.extend_from_slice(&[
            (&tile_program.filter_params_0_uniform,
             UniformData::Vec4(line.from().0.concat_xy_xy(line.vector().0))),
            (&tile_program.filter_params_1_uniform,
             UniformData::Vec4(radii.concat_xy_xy(uv_origin.0))),
            (&tile_program.filter_params_2_uniform, UniformData::Vec4(F32x4::default())),
        ]);
    }

    fn set_uniforms_for_text_filter<'a>(
            &'a self,
            tile_program: &'a TileProgramCommon<D>,
            textures: &mut Vec<TextureBinding<'a, D::TextureParameter, D::Texture>>,
            uniforms: &mut Vec<UniformBinding<'a, D::Uniform>>,
            fg_color: ColorF,
            bg_color: ColorF,
            defringing_kernel: Option<DefringingKernel>,
            gamma_correction: bool) {
        let gamma_lut_texture = self.allocator.get_texture(self.gamma_lut_texture_id);
        textures.push((&tile_program.gamma_lut_texture, gamma_lut_texture));

        match defringing_kernel {
            Some(ref kernel) => {
                uniforms.push((&tile_program.filter_params_0_uniform,
                               UniformData::Vec4(F32x4::from_slice(&kernel.0))));
            }
            None => {
                uniforms.push((&tile_program.filter_params_0_uniform,
                               UniformData::Vec4(F32x4::default())));
            }
        }

        let mut params_2 = fg_color.0;
        params_2.set_w(gamma_correction as i32 as f32);

        uniforms.extend_from_slice(&[
            (&tile_program.filter_params_1_uniform, UniformData::Vec4(bg_color.0)),
            (&tile_program.filter_params_2_uniform, UniformData::Vec4(params_2)),
        ]);
    }

    fn set_uniforms_for_blur_filter<'a>(&'a self,
                                        tile_program: &'a TileProgramCommon<D>,
                                        uniforms: &mut Vec<(&'a D::Uniform, UniformData)>,
                                        direction: BlurDirection,
                                        sigma: f32) {
        let sigma_inv = 1.0 / sigma;
        let gauss_coeff_x = SQRT_2_PI_INV * sigma_inv;
        let gauss_coeff_y = f32::exp(-0.5 * sigma_inv * sigma_inv);
        let gauss_coeff_z = gauss_coeff_y * gauss_coeff_y;

        let src_offset = match direction {
            BlurDirection::X => vec2f(1.0, 0.0),
            BlurDirection::Y => vec2f(0.0, 1.0),
        };

        let support = f32::ceil(1.5 * sigma) * 2.0;

        uniforms.extend_from_slice(&[
            (&tile_program.filter_params_0_uniform,
             UniformData::Vec4(src_offset.0.concat_xy_xy(F32x2::new(support, 0.0)))),
            (&tile_program.filter_params_1_uniform,
             UniformData::Vec4(F32x4::new(gauss_coeff_x, gauss_coeff_y, gauss_coeff_z, 0.0))),
            (&tile_program.filter_params_2_uniform, UniformData::Vec4(F32x4::default())),
        ]);
    }

    // Pattern textures

    fn texture_page(&self, id: TexturePageId) -> &D::Texture {
        self.device.framebuffer_texture(&self.texture_page_framebuffer(id))
    }

    fn texture_page_framebuffer(&self, id: TexturePageId) -> &D::Framebuffer {
        let framebuffer_id = self.pattern_texture_pages[id.0 as usize]
                                 .as_ref()
                                 .expect("Texture page not allocated!")
                                 .framebuffer_id;
        self.allocator.get_framebuffer(framebuffer_id)
    }

    pub(crate) fn clear_color_for_draw_operation(&self) -> Option<ColorF> {
        let must_preserve_contents = match self.render_target_stack.last() {
            Some(&render_target_id) => {
                let texture_page = self.render_target_location(render_target_id).page;
                self.pattern_texture_pages[texture_page.0 as usize]
                    .as_ref()
                    .expect("Draw target texture page not allocated!")
                    .must_preserve_contents
            }
            None => {
                self.framebuffer_flags.contains(FramebufferFlags::DEST_FRAMEBUFFER_IS_DIRTY)
            }
        };

        if must_preserve_contents {
            None
        } else if self.render_target_stack.is_empty() {
            self.options.background_color
        } else {
            Some(ColorF::default())
        }
    }

    // Sizing

    pub(crate) fn tile_size(&self) -> Vector2I {
        let temp = self.draw_viewport().size() +
            vec2i(TILE_WIDTH as i32 - 1, TILE_HEIGHT as i32 - 1);
        vec2i(temp.x() / TILE_WIDTH as i32, temp.y() / TILE_HEIGHT as i32)
    }

    pub(crate) fn framebuffer_tile_size(&self) -> Vector2I {
        pixel_size_to_tile_size(self.dest_framebuffer.window_size(&self.device))
    }

    // Viewport calculation

    fn main_viewport(&self) -> RectI {
        match self.dest_framebuffer {
            DestFramebuffer::Default { viewport, .. } => viewport,
            DestFramebuffer::Other(ref framebuffer) => {
                let texture = self.device.framebuffer_texture(framebuffer);
                let size = self.device.texture_size(texture);
                RectI::new(Vector2I::default(), size)
            }
        }
    }

    fn draw_viewport(&self) -> RectI {
        match self.render_target_stack.last() {
            Some(&render_target_id) => self.render_target_location(render_target_id).rect,
            None => self.main_viewport(),
        }
    }

    pub fn draw_render_target(&self) -> RenderTarget<D> {
        match self.render_target_stack.last() {
            Some(&render_target_id) => {
                let texture_page_id = self.render_target_location(render_target_id).page;
                let framebuffer = self.texture_page_framebuffer(texture_page_id);
                RenderTarget::Framebuffer(framebuffer)
            }
            None => {
                if self.renderer_flags
                       .contains(RendererFlags::INTERMEDIATE_DEST_FRAMEBUFFER_NEEDED) {
                    let intermediate_dest_framebuffer =
                        self.allocator.get_framebuffer(self.intermediate_dest_framebuffer_id);
                    RenderTarget::Framebuffer(intermediate_dest_framebuffer)
                } else {
                    match self.dest_framebuffer {
                        DestFramebuffer::Default { .. } => RenderTarget::Default,
                        DestFramebuffer::Other(ref framebuffer) => {
                            RenderTarget::Framebuffer(framebuffer)
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn preserve_draw_framebuffer(&mut self) {
        match self.render_target_stack.last() {
            Some(&render_target_id) => {
                let texture_page = self.render_target_location(render_target_id).page;
                self.pattern_texture_pages[texture_page.0 as usize]
                    .as_mut()
                    .expect("Draw target texture page not allocated!")
                    .must_preserve_contents = true;
            }
            None => {
                self.framebuffer_flags.insert(FramebufferFlags::DEST_FRAMEBUFFER_IS_DIRTY);
            }
        }
    }

    fn render_target_location(&self, render_target_id: RenderTargetId) -> TextureLocation {
        self.render_targets[render_target_id.render_target as usize].location
    }
}

impl<D> Frame<D> where D: Device {
    // FIXME(pcwalton): This signature shouldn't be so big. Make a struct.
    fn new(device: &D,
           allocator: &mut GPUMemoryAllocator<D>,
           blit_program: &BlitProgram<D>,
           clear_program: &ClearProgram<D>,
           reprojection_program: &ReprojectionProgram<D>,
           stencil_program: &StencilProgram<D>,
           quad_vertex_positions_buffer_id: BufferID,
           quad_vertex_indices_buffer_id: BufferID,
           window_size: Vector2I)
           -> Frame<D> {
        let quad_vertex_positions_buffer = allocator.get_buffer(quad_vertex_positions_buffer_id);
        let quad_vertex_indices_buffer = allocator.get_buffer(quad_vertex_indices_buffer_id);

        let blit_vertex_array = BlitVertexArray::new(device,
                                                     &blit_program,
                                                     &quad_vertex_positions_buffer,
                                                     &quad_vertex_indices_buffer);
        let clear_vertex_array = ClearVertexArray::new(device,
                                                       &clear_program,
                                                       &quad_vertex_positions_buffer,
                                                       &quad_vertex_indices_buffer);
        let reprojection_vertex_array = ReprojectionVertexArray::new(device,
                                                                     &reprojection_program,
                                                                     &quad_vertex_positions_buffer,
                                                                     &quad_vertex_indices_buffer);
        let stencil_vertex_array = StencilVertexArray::new(device, &stencil_program);

        let dest_blend_framebuffer_id = allocator.allocate_framebuffer(device,
                                                                       window_size,
                                                                       TextureFormat::RGBA8,
                                                                       FramebufferTag::DestBlend);

        Frame {
            blit_vertex_array,
            clear_vertex_array,
            reprojection_vertex_array,
            stencil_vertex_array,
            quads_vertex_indices_buffer_id: None,
            quads_vertex_indices_length: 0,
            buffered_fills: vec![],
            pending_fills: vec![],
            mask_temp_framebuffer: None,
            dest_blend_framebuffer_id,
        }
    }
}

impl<D> RendererLevelImpl<D> where D: Device {
    #[inline]
    fn require_d3d11(&mut self) -> &mut RendererD3D11<D> {
        match *self {
            RendererLevelImpl::D3D11(ref mut d3d11_renderer) => d3d11_renderer,
        }
    }
}

#[derive(Clone)]
pub(crate) struct TileBatchInfoD3D9 {
    pub(crate) tile_count: u32,
    pub(crate) z_buffer_id: BufferID,
    tile_vertex_buffer_id: BufferID,
}

// Render stats

bitflags! {
    pub(crate) struct FramebufferFlags: u8 {
        const MASK_FRAMEBUFFER_IS_DIRTY = 0x01;
        const DEST_FRAMEBUFFER_IS_DIRTY = 0x02;
    }
}

struct RenderTargetInfo {
    location: TextureLocation,
}

trait ToBlendState {
    fn to_blend_state(self) -> Option<BlendState>;
}

impl ToBlendState for BlendMode {
    fn to_blend_state(self) -> Option<BlendState> {
        match self {
            BlendMode::Clear => {
                Some(BlendState {
                    src_rgb_factor: BlendFactor::Zero,
                    dest_rgb_factor: BlendFactor::Zero,
                    src_alpha_factor: BlendFactor::Zero,
                    dest_alpha_factor: BlendFactor::Zero,
                    ..BlendState::default()
                })
            }
            BlendMode::SrcOver => {
                Some(BlendState {
                    src_rgb_factor: BlendFactor::One,
                    dest_rgb_factor: BlendFactor::OneMinusSrcAlpha,
                    src_alpha_factor: BlendFactor::One,
                    dest_alpha_factor: BlendFactor::OneMinusSrcAlpha,
                    ..BlendState::default()
                })
            }
            BlendMode::DestOver => {
                Some(BlendState {
                    src_rgb_factor: BlendFactor::OneMinusDestAlpha,
                    dest_rgb_factor: BlendFactor::One,
                    src_alpha_factor: BlendFactor::OneMinusDestAlpha,
                    dest_alpha_factor: BlendFactor::One,
                    ..BlendState::default()
                })
            }
            BlendMode::SrcIn => {
                Some(BlendState {
                    src_rgb_factor: BlendFactor::DestAlpha,
                    dest_rgb_factor: BlendFactor::Zero,
                    src_alpha_factor: BlendFactor::DestAlpha,
                    dest_alpha_factor: BlendFactor::Zero,
                    ..BlendState::default()
                })
            }
            BlendMode::DestIn => {
                Some(BlendState {
                    src_rgb_factor: BlendFactor::Zero,
                    dest_rgb_factor: BlendFactor::SrcAlpha,
                    src_alpha_factor: BlendFactor::Zero,
                    dest_alpha_factor: BlendFactor::SrcAlpha,
                    ..BlendState::default()
                })
            }
            BlendMode::SrcOut => {
                Some(BlendState {
                    src_rgb_factor: BlendFactor::OneMinusDestAlpha,
                    dest_rgb_factor: BlendFactor::Zero,
                    src_alpha_factor: BlendFactor::OneMinusDestAlpha,
                    dest_alpha_factor: BlendFactor::Zero,
                    ..BlendState::default()
                })
            }
            BlendMode::DestOut => {
                Some(BlendState {
                    src_rgb_factor: BlendFactor::Zero,
                    dest_rgb_factor: BlendFactor::OneMinusSrcAlpha,
                    src_alpha_factor: BlendFactor::Zero,
                    dest_alpha_factor: BlendFactor::OneMinusSrcAlpha,
                    ..BlendState::default()
                })
            }
            BlendMode::SrcAtop => {
                Some(BlendState {
                    src_rgb_factor: BlendFactor::DestAlpha,
                    dest_rgb_factor: BlendFactor::OneMinusSrcAlpha,
                    src_alpha_factor: BlendFactor::DestAlpha,
                    dest_alpha_factor: BlendFactor::OneMinusSrcAlpha,
                    ..BlendState::default()
                })
            }
            BlendMode::DestAtop => {
                Some(BlendState {
                    src_rgb_factor: BlendFactor::OneMinusDestAlpha,
                    dest_rgb_factor: BlendFactor::SrcAlpha,
                    src_alpha_factor: BlendFactor::OneMinusDestAlpha,
                    dest_alpha_factor: BlendFactor::SrcAlpha,
                    ..BlendState::default()
                })
            }
            BlendMode::Xor => {
                Some(BlendState {
                    src_rgb_factor: BlendFactor::OneMinusDestAlpha,
                    dest_rgb_factor: BlendFactor::OneMinusSrcAlpha,
                    src_alpha_factor: BlendFactor::OneMinusDestAlpha,
                    dest_alpha_factor: BlendFactor::OneMinusSrcAlpha,
                    ..BlendState::default()
                })
            }
            BlendMode::Lighter => {
                Some(BlendState {
                    src_rgb_factor: BlendFactor::One,
                    dest_rgb_factor: BlendFactor::One,
                    src_alpha_factor: BlendFactor::One,
                    dest_alpha_factor: BlendFactor::One,
                    ..BlendState::default()
                })
            }
            BlendMode::Copy |
            BlendMode::Darken |
            BlendMode::Lighten |
            BlendMode::Multiply |
            BlendMode::Screen |
            BlendMode::HardLight |
            BlendMode::Overlay |
            BlendMode::ColorDodge |
            BlendMode::ColorBurn |
            BlendMode::SoftLight |
            BlendMode::Difference |
            BlendMode::Exclusion |
            BlendMode::Hue |
            BlendMode::Saturation |
            BlendMode::Color |
            BlendMode::Luminosity => {
                // Blending is done manually in the shader.
                None
            }
        }
    }
}

pub trait BlendModeExt {
    fn needs_readable_framebuffer(self) -> bool;
}

impl BlendModeExt for BlendMode {
    fn needs_readable_framebuffer(self) -> bool {
        match self {
            BlendMode::Clear |
            BlendMode::SrcOver |
            BlendMode::DestOver |
            BlendMode::SrcIn |
            BlendMode::DestIn |
            BlendMode::SrcOut |
            BlendMode::DestOut |
            BlendMode::SrcAtop |
            BlendMode::DestAtop |
            BlendMode::Xor |
            BlendMode::Lighter |
            BlendMode::Copy => false,
            BlendMode::Lighten |
            BlendMode::Darken |
            BlendMode::Multiply |
            BlendMode::Screen |
            BlendMode::HardLight |
            BlendMode::Overlay |
            BlendMode::ColorDodge |
            BlendMode::ColorBurn |
            BlendMode::SoftLight |
            BlendMode::Difference |
            BlendMode::Exclusion |
            BlendMode::Hue |
            BlendMode::Saturation |
            BlendMode::Color |
            BlendMode::Luminosity => true,
        }
    }
}

bitflags! {
    pub(crate) struct RendererFlags: u8 {
        // Whether we need a depth buffer.
        const USE_DEPTH = 0x01;
        // Whether an intermediate destination framebuffer is needed.
        //
        // This will be true if any exotic blend modes are used at the top level (not inside a
        // render target), *and* the output framebuffer is the default framebuffer.
        const INTERMEDIATE_DEST_FRAMEBUFFER_NEEDED = 0x02;
    }
}

trait ToCompositeCtrl {
    fn to_composite_ctrl(&self) -> i32;
}

impl ToCompositeCtrl for BlendMode {
    fn to_composite_ctrl(&self) -> i32 {
        match *self {
            BlendMode::SrcOver |
            BlendMode::SrcAtop |
            BlendMode::DestOver |
            BlendMode::DestOut |
            BlendMode::Xor |
            BlendMode::Lighter |
            BlendMode::Clear |
            BlendMode::Copy |
            BlendMode::SrcIn |
            BlendMode::SrcOut |
            BlendMode::DestIn |
            BlendMode::DestAtop => COMBINER_CTRL_COMPOSITE_NORMAL,
            BlendMode::Multiply => COMBINER_CTRL_COMPOSITE_MULTIPLY,
            BlendMode::Darken => COMBINER_CTRL_COMPOSITE_DARKEN,
            BlendMode::Lighten => COMBINER_CTRL_COMPOSITE_LIGHTEN,
            BlendMode::Screen => COMBINER_CTRL_COMPOSITE_SCREEN,
            BlendMode::Overlay => COMBINER_CTRL_COMPOSITE_OVERLAY,
            BlendMode::ColorDodge => COMBINER_CTRL_COMPOSITE_COLOR_DODGE,
            BlendMode::ColorBurn => COMBINER_CTRL_COMPOSITE_COLOR_BURN,
            BlendMode::HardLight => COMBINER_CTRL_COMPOSITE_HARD_LIGHT,
            BlendMode::SoftLight => COMBINER_CTRL_COMPOSITE_SOFT_LIGHT,
            BlendMode::Difference => COMBINER_CTRL_COMPOSITE_DIFFERENCE,
            BlendMode::Exclusion => COMBINER_CTRL_COMPOSITE_EXCLUSION,
            BlendMode::Hue => COMBINER_CTRL_COMPOSITE_HUE,
            BlendMode::Saturation => COMBINER_CTRL_COMPOSITE_SATURATION,
            BlendMode::Color => COMBINER_CTRL_COMPOSITE_COLOR,
            BlendMode::Luminosity => COMBINER_CTRL_COMPOSITE_LUMINOSITY,
        }
    }
}

trait ToCombineMode {
    fn to_combine_mode(self) -> i32;
}

impl ToCombineMode for PaintCompositeOp {
    fn to_combine_mode(self) -> i32 {
        match self {
            PaintCompositeOp::DestIn => COMBINER_CTRL_COLOR_COMBINE_DEST_IN,
            PaintCompositeOp::SrcIn => COMBINER_CTRL_COLOR_COMBINE_SRC_IN,
        }
    }
}

fn pixel_size_to_tile_size(pixel_size: Vector2I) -> Vector2I {
    // Round up.
    let tile_size = vec2i(TILE_WIDTH as i32 - 1, TILE_HEIGHT as i32 - 1);
    let size = pixel_size + tile_size;
    vec2i(size.x() / TILE_WIDTH as i32, size.y() / TILE_HEIGHT as i32)
}

#[derive(Clone)]
struct FillBufferInfoD3D9 {
    fill_buffer_id: BufferID,
    fill_count: u32,
}
