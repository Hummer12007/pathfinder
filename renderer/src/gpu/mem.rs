// pathfinder/renderer/src/gpu/mem.rs
//
// Copyright © 2020 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! GPU memory management.

use crate::gpu::options::RendererLevel;
use crate::tiles::{TILE_HEIGHT, TILE_WIDTH};
use fxhash::FxHashMap;
use pathfinder_geometry::vector::{Vector2I, vec2i};
use pathfinder_gpu::{BufferData, BufferTarget, BufferUploadMode, Device};
use pathfinder_gpu::{TextureFormat, TextureSamplingFlags};
use std::collections::VecDeque;
use std::default::Default;
use std::mem;

const MIN_SIZE_CLASS: usize = 4;                    // 16 bytes
const MAX_GPU_MEMORY_USAGE: u64 = 64 * 1024 * 1024; // 64 MB

pub(crate) struct GPUMemoryAllocator<D> where D: Device {
    buffers_in_use: FxHashMap<BufferID, BufferAllocation<D>>,
    textures_in_use: FxHashMap<TextureID, TextureAllocation<D>>,
    framebuffers_in_use: FxHashMap<FramebufferID, FramebufferAllocation<D>>,
    free_buffers: VecDeque<FreeBuffer<D>>,
    next_buffer_id: BufferID,
    next_texture_id: TextureID,
    next_framebuffer_id: FramebufferID,
    bytes_committed: u64,
    bytes_allocated: u64,
}

struct BufferAllocation<D> where D: Device {
    buffer: D::Buffer,
    size: u64,
    tag: BufferTag,
}

struct TextureAllocation<D> where D: Device {
    texture: D::Texture,
    descriptor: TextureDescriptor,
    tag: TextureTag,
}

struct FramebufferAllocation<D> where D: Device {
    framebuffer: D::Framebuffer,
    descriptor: TextureDescriptor,
    tag: FramebufferTag,
}

struct FreeBuffer<D> where D: Device {
    id: BufferID,
    allocation: BufferAllocation<D>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct TextureDescriptor {
    width: u32,
    height: u32,
    format: TextureFormat,
}

// ID of any resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ResourceID {
    Buffer(BufferID),
    Texture(TextureID),
    Framebuffer(FramebufferID),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BufferID(pub(crate) u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct TextureID(pub(crate) u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct FramebufferID(pub(crate) u64);

// For debugging and profiling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum BufferTag {
    QuadVertexPositions,
    QuadVertexIndices,
    QuadsVertexIndices,
    Fill,
    TileD3D11,
    TilePathInfoD3D11,
    PropagateMetadataD3D11,
    BackdropInfoD3D11,
    MicrolineD3D11,
    DiceMetadataD3D11,
    DiceIndirectDrawParamsD3D11,
    FillIndirectDrawParamsD3D11,
    ZBufferD3D11,
    FirstTileD3D11,
    AlphaTileD3D11,
    PointsD3D11,
    PointIndicesD3D11,
}

// For debugging and profiling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TextureTag {
    AreaLUT,
    GammaLUT,
    TextureMetadata,
}

// For debugging and profiling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FramebufferTag {
    TileAlphaMask,
    PatternPage,
    IntermediateDest,
    DestBlend,
}

impl<D> GPUMemoryAllocator<D> where D: Device {
    pub(crate) fn new() -> GPUMemoryAllocator<D> {
        GPUMemoryAllocator {
            buffers_in_use: FxHashMap::default(),
            textures_in_use: FxHashMap::default(),
            framebuffers_in_use: FxHashMap::default(),
            free_buffers: VecDeque::new(),
            next_buffer_id: BufferID(0),
            next_texture_id: TextureID(0),
            next_framebuffer_id: FramebufferID(0),
            bytes_committed: 0,
            bytes_allocated: 0,
        }
    }

    pub(crate) fn allocate_buffer<T>(&mut self, device: &D, size: u64, tag: BufferTag)
                                     -> BufferID {
        // TODO(pcwalton): This should be smarter! Use size classes!
        let byte_size = size * mem::size_of::<T>() as u64;
        for free_buffer_index in 0..self.free_buffers.len() {
            if self.free_buffers[free_buffer_index].allocation.size == byte_size {
                let mut free_buffer = self.free_buffers.remove(free_buffer_index).unwrap();
                free_buffer.allocation.tag = tag;
                self.bytes_committed += free_buffer.allocation.size;
                self.buffers_in_use.insert(free_buffer.id, free_buffer.allocation);
                return free_buffer.id;
            }
        }

        let buffer = device.create_buffer(BufferUploadMode::Dynamic);
        device.allocate_buffer::<T>(&buffer,
                                     BufferData::Uninitialized(size as usize),
                                     BufferTarget::Storage);
        let id = self.next_buffer_id;
        self.next_buffer_id.0 += 1;

        self.buffers_in_use.insert(id, BufferAllocation { buffer, size: byte_size, tag });
        self.bytes_allocated += byte_size;
        self.bytes_committed += byte_size;

        id
    }

    pub(crate) fn allocate_texture(&mut self,
                                   device: &D,
                                   size: Vector2I,
                                   format: TextureFormat,
                                   tag: TextureTag)
                                   -> TextureID {
        let texture = device.create_texture(format, size);
        let id = self.next_texture_id;
        self.next_texture_id.0 += 1;

        let descriptor = TextureDescriptor {
            width: size.x() as u32,
            height: size.y() as u32,
            format,
        };
        self.textures_in_use.insert(id, TextureAllocation { texture, descriptor, tag });

        let byte_size = descriptor.byte_size();
        self.bytes_allocated += byte_size;
        self.bytes_committed += byte_size;

        id
    }

    pub(crate) fn allocate_framebuffer(&mut self,
                                       device: &D,
                                       size: Vector2I,
                                       format: TextureFormat,
                                       tag: FramebufferTag)
                                       -> FramebufferID {
        let texture = device.create_texture(format, size);
        let framebuffer = device.create_framebuffer(texture);
        let id = self.next_framebuffer_id;
        self.next_framebuffer_id.0 += 1;

        let descriptor = TextureDescriptor {
            width: size.x() as u32,
            height: size.y() as u32,
            format,
        };
        self.framebuffers_in_use.insert(id, FramebufferAllocation {
            framebuffer,
            descriptor,
            tag,
        });

        let byte_size = descriptor.byte_size();
        self.bytes_allocated += byte_size;
        self.bytes_committed += byte_size;

        id
    }

    pub(crate) fn free_buffer(&mut self, id: BufferID) {
        let allocation = self.buffers_in_use
                             .remove(&id)
                             .expect("Attempted to free unallocated buffer!");
        self.bytes_committed -= allocation.size;
        self.free_buffers.push_back(FreeBuffer { id, allocation });

        // Trim memory if needed.
        while self.bytes_allocated > MAX_GPU_MEMORY_USAGE {
            match self.free_buffers.pop_front() {
                None => break,
                Some(buffer_to_purge) => self.bytes_allocated -= buffer_to_purge.allocation.size,
            }
        }
    }

    pub(crate) fn free_texture(&mut self, id: TextureID) {
        let allocation = self.textures_in_use
                             .remove(&id)
                             .expect("Attempted to free unallocated texture!");
        let byte_size = allocation.descriptor.byte_size();
        self.bytes_allocated -= byte_size;
        self.bytes_committed -= byte_size;
    }

    pub(crate) fn free_framebuffer(&mut self, id: FramebufferID) {
        let allocation = self.framebuffers_in_use
                             .remove(&id)
                             .expect("Attempted to free unallocated framebuffer!");
        let byte_size = allocation.descriptor.byte_size();
        self.bytes_allocated -= byte_size;
        self.bytes_committed -= byte_size;
    }

    pub(crate) fn get_buffer(&self, id: BufferID) -> &D::Buffer {
        &self.buffers_in_use[&id].buffer
    }

    pub(crate) fn get_texture(&self, id: TextureID) -> &D::Texture {
        &self.textures_in_use[&id].texture
    }

    pub(crate) fn get_framebuffer(&self, id: FramebufferID) -> &D::Framebuffer {
        &self.framebuffers_in_use[&id].framebuffer
    }

    #[inline]
    pub(crate) fn bytes_allocated(&self) -> u64 {
        self.bytes_allocated
    }

    #[inline]
    pub(crate) fn bytes_committed(&self) -> u64 {
        self.bytes_committed
    }

    pub(crate) fn dump(&self) {
        println!("GPU memory dump");
        println!("---------------");

        println!("Buffers:");
        let mut ids: Vec<BufferID> = self.buffers_in_use.keys().cloned().collect();
        ids.sort();
        for id in ids {
            let allocation = &self.buffers_in_use[&id];
            println!("id {:?}: {:?} ({:?} B)", id, allocation.tag, allocation.size);
        }

        println!("Textures:");
        let mut ids: Vec<TextureID> = self.textures_in_use.keys().cloned().collect();
        ids.sort();
        for id in ids {
            let allocation = &self.textures_in_use[&id];
            println!("id {:?}: {:?} {:?}x{:?} {:?} ({:?} B)",
                     id,
                     allocation.tag,
                     allocation.descriptor.width,
                     allocation.descriptor.height,
                     allocation.descriptor.format,
                     allocation.descriptor.byte_size());
        }

        println!("Framebuffers:");
        let mut ids: Vec<FramebufferID> = self.framebuffers_in_use.keys().cloned().collect();
        ids.sort();
        for id in ids {
            let allocation = &self.framebuffers_in_use[&id];
            println!("id {:?}: {:?} {:?}x{:?} {:?} ({:?} B)",
                     id,
                     allocation.tag,
                     allocation.descriptor.width,
                     allocation.descriptor.height,
                     allocation.descriptor.format,
                     allocation.descriptor.byte_size());
        }
    }
}

impl TextureDescriptor {
    fn byte_size(&self) -> u64 {
        self.width as u64 * self.height as u64 * self.format.bytes_per_pixel() as u64
    }
}

/*

// Old allocator:

const TEXTURE_CACHE_SIZE: usize = 8;

const MIN_PATH_INFO_STORAGE_CLASS:               usize = 10;    // 1024 entries
const MIN_DICE_METADATA_STORAGE_CLASS:           usize = 10;    // 1024 entries
const MIN_FILL_STORAGE_CLASS:                    usize = 14;    // 16K entries, 128kB
const MIN_TILE_STORAGE_CLASS:                    usize = 10;    // 1024 entries, 12kB
const MIN_TILE_PROPAGATE_METADATA_STORAGE_CLASS: usize = 8;     // 256 entries
const MIN_FIRST_TILE_MAP_STORAGE_CLASS:          usize = 12;    // 4096 entries
const MIN_CLIP_VERTEX_STORAGE_CLASS:             usize = 10;    // 1024 entries, 16kB
const MIN_ALPHA_TILES_STORAGE_CLASS:             usize = 12;    // 4096 entries, 16kB
const MIN_BACKDROPS_STORAGE_CLASS:               usize = 12;    // 4096 entries
const MIN_MICROLINES_STORAGE_CLASS:              usize = 14;    // 16K entries
const MIN_TILES_D3D11_STORAGE_CLASS:             usize = 10;    // 1024 entries, 20kB

pub(crate) struct StorageAllocators<D> where D: Device {
    pub(crate) path_info: StorageAllocator<StorageBuffer<D, TilePathInfo>>,
    pub(crate) dice_metadata: StorageAllocator<DiceMetadataStorage<D>>,
    pub(crate) fill_vertex: StorageAllocator<FillVertexStorage<D>>,
    pub(crate) tile_vertex: StorageAllocator<TileVertexStorage<D>>,
    pub(crate) tile_propagate_metadata: StorageAllocator<StorageBuffer<D, PropagateMetadata>>,
    pub(crate) clip_vertex: StorageAllocator<ClipVertexStorage<D>>,
    pub(crate) first_tile_map: StorageAllocator<StorageBuffer<D, FirstTile>>,
    pub(crate) alpha_tiles: StorageAllocator<StorageBuffer<D, AlphaTileD3D11>>,
    pub(crate) backdrops: StorageAllocator<StorageBuffer<D, BackdropInfo>>,
    pub(crate) microlines: StorageAllocator<StorageBuffer<D, Microline>>,
    pub(crate) tiles_d3d11: StorageAllocator<StorageBuffer<D, TileD3D11>>,
    pub(crate) z_buffers: ZBufferStorageAllocator<D>,
}

pub(crate) trait Storage {
    fn gpu_bytes_allocated(&self) -> u64;
}

pub(crate) struct StorageAllocator<S> where S: Storage {
    buckets: Vec<StorageAllocatorBucket<S>>,
    min_size_class: usize,
}

struct StorageAllocatorBucket<S> {
    free: Vec<S>,
    in_use: Vec<S>,
}

pub(crate) struct ZBufferStorageAllocator<D> where D: Device {
    bucket: StorageAllocatorBucket<ZBuffer<D>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StorageID {
    bucket: usize,
    index: usize,
}

impl<D> StorageAllocators<D> where D: Device {
    pub(crate) fn new() -> StorageAllocators<D> {
        let path_info = StorageAllocator::new(MIN_PATH_INFO_STORAGE_CLASS);
        let dice_metadata = StorageAllocator::new(MIN_DICE_METADATA_STORAGE_CLASS);
        let fill_vertex = StorageAllocator::new(MIN_FILL_STORAGE_CLASS);
        let tile_vertex = StorageAllocator::new(MIN_TILE_STORAGE_CLASS);
        let tile_propagate_metadata =
            StorageAllocator::new(MIN_TILE_PROPAGATE_METADATA_STORAGE_CLASS);
        let clip_vertex = StorageAllocator::new(MIN_CLIP_VERTEX_STORAGE_CLASS);
        let first_tile_map = StorageAllocator::new(MIN_FIRST_TILE_MAP_STORAGE_CLASS);
        let alpha_tiles = StorageAllocator::new(MIN_ALPHA_TILES_STORAGE_CLASS);
        let backdrops = StorageAllocator::new(MIN_BACKDROPS_STORAGE_CLASS);
        let microlines = StorageAllocator::new(MIN_MICROLINES_STORAGE_CLASS);
        let tiles_d3d11 = StorageAllocator::new(MIN_TILES_D3D11_STORAGE_CLASS);
        let z_buffers = ZBufferStorageAllocator::new();

        StorageAllocators {
            path_info,
            dice_metadata,
            fill_vertex,
            tile_vertex,
            tile_propagate_metadata,
            clip_vertex,
            first_tile_map,
            alpha_tiles,
            backdrops,
            microlines,
            tiles_d3d11,
            z_buffers,
        }
    }

    pub(crate) fn end_frame(&mut self) {
        self.path_info.end_frame();
        self.dice_metadata.end_frame();
        self.fill_vertex.end_frame();
        self.tile_vertex.end_frame();
        self.tile_propagate_metadata.end_frame();
        self.clip_vertex.end_frame();
        self.first_tile_map.end_frame();
        self.alpha_tiles.end_frame();
        self.backdrops.end_frame();
        self.microlines.end_frame();
        self.tiles_d3d11.end_frame();
        self.z_buffers.end_frame();
    }

    pub(crate) fn gpu_bytes_allocated(&self) -> u64 {
        self.path_info.gpu_bytes_allocated() +
            self.dice_metadata.gpu_bytes_allocated() +
            self.fill_vertex.gpu_bytes_allocated() +
            self.tile_vertex.gpu_bytes_allocated() +
            self.tile_propagate_metadata.gpu_bytes_allocated() +
            self.clip_vertex.gpu_bytes_allocated() +
            self.first_tile_map.gpu_bytes_allocated() +
            self.alpha_tiles.gpu_bytes_allocated() +
            self.backdrops.gpu_bytes_allocated() +
            self.microlines.gpu_bytes_allocated() +
            self.tiles_d3d11.gpu_bytes_allocated() +
            self.z_buffers.gpu_bytes_allocated()
    }

    #[allow(dead_code)]
    fn dump(&self) {
        println!("path_info {}", self.path_info.gpu_bytes_allocated());
        println!("dice_metadata {}", self.dice_metadata.gpu_bytes_allocated());
        println!("fill_vertex {}", self.fill_vertex.gpu_bytes_allocated());
        println!("tile_vertex {}", self.tile_vertex.gpu_bytes_allocated());
        println!("tile_propagate_metadata {}", self.tile_propagate_metadata.gpu_bytes_allocated());
        println!("clip_vertex {}", self.clip_vertex.gpu_bytes_allocated());
        println!("first_tile_map {}", self.first_tile_map.gpu_bytes_allocated());
        println!("alpha_tiles {}", self.alpha_tiles.gpu_bytes_allocated());
        println!("backdrops {}", self.backdrops.gpu_bytes_allocated());
        println!("microlines {}", self.microlines.gpu_bytes_allocated());
        println!("tiles_d3d11 {}", self.tiles_d3d11.gpu_bytes_allocated());
        println!("z_buffers {}", self.z_buffers.gpu_bytes_allocated());
    }
}

impl<S> StorageAllocator<S> where S: Storage {
    fn new(min_size_class: usize) -> StorageAllocator<S> {
        StorageAllocator { buckets: vec![], min_size_class }
    }

    pub(crate) fn allocate<F>(&mut self, size: u64, allocator: F) -> StorageID
                              where F: FnOnce(u64) -> S {
        let size_class = (64 - (size.leading_zeros() as usize)).max(self.min_size_class);
        let bucket_index = size_class - self.min_size_class;
        while self.buckets.len() < bucket_index + 1 {
            self.buckets.push(StorageAllocatorBucket::new());
        }

        let bucket = &mut self.buckets[bucket_index];
        match bucket.free.pop() {
            Some(storage) => bucket.in_use.push(storage),
            None => bucket.in_use.push(allocator(1 << size_class as u64)),
        }
        StorageID { bucket: bucket_index, index: bucket.in_use.len() - 1 }
    }

    pub(crate) fn get(&self, storage_id: StorageID) -> &S {
        &self.buckets[storage_id.bucket].in_use[storage_id.index]
    }

    pub(crate) fn end_frame(&mut self) {
        self.buckets.iter_mut().for_each(|bucket| bucket.end_frame());
    }

    fn gpu_bytes_allocated(&self) -> u64 {
        let mut total = 0;
        for bucket in &self.buckets {
            total += bucket.gpu_bytes_allocated();
        }
        total
    }
}

impl<D, T> StorageAllocator<StorageBuffer<D, T>> where D: Device {
    pub(crate) fn allocate_buffer(&mut self, device: &D, size: u64, target: BufferTarget)
                                  -> StorageID {
        self.allocate(size, |size| StorageBuffer::allocate(device, size, target))
    }
}

impl<S> StorageAllocatorBucket<S> where S: Storage {
    fn new() -> StorageAllocatorBucket<S> {
        StorageAllocatorBucket { free: vec![], in_use: vec![] }
    }

    fn end_frame(&mut self) {
        self.free.extend(mem::replace(&mut self.in_use, vec![]).into_iter())
    }

    fn gpu_bytes_allocated(&self) -> u64 {
        let mut total = 0;
        for storage in &self.free {
            total += storage.gpu_bytes_allocated();
        }
        for storage in &self.in_use {
            total += storage.gpu_bytes_allocated();
        }
        total
    }
}

impl<D> ZBufferStorageAllocator<D> where D: Device {
    fn new() -> ZBufferStorageAllocator<D> {
        ZBufferStorageAllocator { bucket: StorageAllocatorBucket::new() }
    }

    pub(crate) fn allocate(&mut self,
                           device: &D,
                           renderer_level: RendererLevel,
                           framebuffer_size: Vector2I)
                           -> StorageID {
        match self.bucket.free.pop() {
            Some(storage) => self.bucket.in_use.push(storage),
            None => {
                self.bucket.in_use.push(ZBuffer::new(device, renderer_level, framebuffer_size))
            }
        }
        StorageID { bucket: 0, index: self.bucket.in_use.len() - 1 }
    }

    pub(crate) fn get(&self, storage_id: StorageID) -> &ZBuffer<D> {
        &self.bucket.in_use[storage_id.index]
    }

    pub(crate) fn end_frame(&mut self) {
        self.bucket.end_frame()
    }

    fn gpu_bytes_allocated(&self) -> u64 {
        self.bucket.gpu_bytes_allocated()
    }
}

pub(crate) struct StorageBuffer<D, T> where D: Device {
    pub(crate) buffer: D::Buffer,
    pub(crate) size: u64,
    phantom: PhantomData<T>,
}

impl<D, T> Storage for StorageBuffer<D, T> where D: Device {
    fn gpu_bytes_allocated(&self) -> u64 {
        self.size
    }
}

impl<D, T> StorageBuffer<D, T> where D: Device {
    pub(crate) fn allocate(device: &D, size: u64, target: BufferTarget) -> StorageBuffer<D, T> {
        let buffer = device.create_buffer(BufferUploadMode::Dynamic);
        device.allocate_buffer::<T>(&buffer, BufferData::Uninitialized(size as usize), target);
        StorageBuffer {
            buffer,
            size: size * mem::size_of::<T>() as u64,
            phantom: PhantomData,
        }
    }
}

pub(crate) struct DiceMetadataStorage<D> where D: Device {
    pub(crate) metadata_buffer: D::Buffer,
    pub(crate) indirect_draw_params_buffer: D::Buffer,
    pub(crate) size: u64,
}

pub(crate) struct FillVertexStorage<D> where D: Device {
    pub(crate) vertex_buffer: D::Buffer,
    // Will be `None` if we're using compute.
    pub(crate) vertex_array: Option<FillVertexArray<D>>,
    pub(crate) indirect_draw_params_buffer: Option<D::Buffer>,
    pub(crate) size: u64,
}

pub(crate) struct TileVertexStorage<D> where D: Device {
    pub(crate) tile_vertex_array: Option<TileVertexArray<D>>,
    pub(crate) tile_copy_vertex_array: CopyTileVertexArray<D>,
    pub(crate) vertex_buffer: D::Buffer,
    pub(crate) size: u64,
}

pub(crate) struct ClipVertexStorage<D> where D: Device {
    pub(crate) tile_clip_copy_vertex_array: ClipTileCopyVertexArray<D>,
    pub(crate) tile_clip_combine_vertex_array: ClipTileCombineVertexArray<D>,
    pub(crate) vertex_buffer: D::Buffer,
    pub(crate) size: u64,
}

*/

/*

impl<D> DiceMetadataStorage<D> where D: Device {
    pub(crate) fn new(device: &D, size: u64) -> DiceMetadataStorage<D> {
        let metadata_buffer = device.create_buffer(BufferUploadMode::Dynamic);
        let indirect_draw_params_buffer = device.create_buffer(BufferUploadMode::Dynamic);
        device.allocate_buffer::<DiceMetadata>(&metadata_buffer,
                                               BufferData::Uninitialized(size as usize),
                                               BufferTarget::Storage);
        device.allocate_buffer::<u32>(&indirect_draw_params_buffer,
                                      BufferData::Uninitialized(8),
                                      BufferTarget::Storage);
        DiceMetadataStorage { metadata_buffer, indirect_draw_params_buffer, size }
    }
}

impl<D> Storage for DiceMetadataStorage<D> where D: Device {
    fn gpu_bytes_allocated(&self) -> u64 {
        self.size * (mem::size_of::<DiceMetadata>() as u64 + mem::size_of::<u32>() as u64)
    }
}

impl<D> FillVertexStorage<D> where D: Device {
    pub(crate) fn new(size: u64,
                      device: &D,
                      fill_program: &FillProgram<D>,
                      quad_vertex_positions_buffer: &D::Buffer,
                      quad_vertex_indices_buffer: &D::Buffer,
                      renderer_level: RendererLevel)
                      -> FillVertexStorage<D> {
        let vertex_buffer = device.create_buffer(BufferUploadMode::Dynamic);
        let vertex_buffer_data: BufferData<Fill> = BufferData::Uninitialized(size as usize);
        device.allocate_buffer(&vertex_buffer, vertex_buffer_data, BufferTarget::Vertex);

        let vertex_array = match *fill_program {
            FillProgram::Raster(ref fill_raster_program) => {
                Some(FillVertexArray::new(device,
                                          fill_raster_program,
                                          &vertex_buffer,
                                          quad_vertex_positions_buffer,
                                          quad_vertex_indices_buffer))
            }
            FillProgram::Compute(_) => None,
        };

        let indirect_draw_params_buffer = match renderer_level {
            RendererLevel::D3D11 => {
                let indirect_draw_params_buffer = device.create_buffer(BufferUploadMode::Static);
                device.allocate_buffer::<u32>(&indirect_draw_params_buffer,
                                              BufferData::Uninitialized(8),
                                              BufferTarget::Storage);
                Some(indirect_draw_params_buffer)
            }
            RendererLevel::D3D9 => None,
        };

        FillVertexStorage { vertex_buffer, vertex_array, indirect_draw_params_buffer, size }
    }
}

impl<D> Storage for FillVertexStorage<D> where D: Device {
    fn gpu_bytes_allocated(&self) -> u64 {
        let mut total = self.size * mem::size_of::<Fill>() as u64;
        if self.indirect_draw_params_buffer.is_some() {
            total += 8;
        }
        total
    }
}

impl<D> TileVertexStorage<D> where D: Device {
    pub(crate) fn new(size: u64,
                      device: &D,
                      tile_program: &TileProgram<D>,
                      tile_copy_program: &CopyTileProgram<D>,
                      quad_vertex_positions_buffer: &D::Buffer,
                      quad_vertex_indices_buffer: &D::Buffer)
                      -> TileVertexStorage<D> {
        let vertex_buffer = device.create_buffer(BufferUploadMode::Dynamic);
        device.allocate_buffer::<TileObjectPrimitive>(&vertex_buffer,
                                                      BufferData::Uninitialized(size as usize),
                                                      BufferTarget::Vertex);
        let tile_vertex_array = match *tile_program {
            TileProgram::Compute(_) => None,
            TileProgram::Raster(ref tile_raster_program) => {
                Some(TileVertexArray::new(device,
                                          tile_raster_program,
                                          &vertex_buffer,
                                          &quad_vertex_positions_buffer,
                                          &quad_vertex_indices_buffer))
            }
        };
        let tile_copy_vertex_array = CopyTileVertexArray::new(device,
                                                              &tile_copy_program,
                                                              &vertex_buffer,
                                                              &quad_vertex_indices_buffer);
        TileVertexStorage {
            vertex_buffer,
            tile_vertex_array,
            tile_copy_vertex_array,
            size,
        }
    }
}

impl<D> Storage for TileVertexStorage<D> where D: Device {
    fn gpu_bytes_allocated(&self) -> u64 {
        self.size * mem::size_of::<TileObjectPrimitive>() as u64
    }
}

impl<D> ClipVertexStorage<D> where D: Device {
    pub(crate) fn new(size: u64,
                      device: &D,
                      tile_clip_combine_program: &ClipTileCombineProgram<D>,
                      tile_clip_copy_program: &ClipTileCopyProgram<D>,
                      quad_vertex_positions_buffer: &D::Buffer,
                      quad_vertex_indices_buffer: &D::Buffer)
                      -> ClipVertexStorage<D> {
        let vertex_buffer = device.create_buffer(BufferUploadMode::Dynamic);
        device.allocate_buffer::<Clip>(&vertex_buffer,
                                       BufferData::Uninitialized(size as usize),
                                       BufferTarget::Vertex);
        let tile_clip_combine_vertex_array =
            ClipTileCombineVertexArray::new(device,
                                            &tile_clip_combine_program,
                                            &vertex_buffer,
                                            &quad_vertex_positions_buffer,
                                            &quad_vertex_indices_buffer);
        let tile_clip_copy_vertex_array =
            ClipTileCopyVertexArray::new(device,
                                         &tile_clip_copy_program,
                                         &vertex_buffer,
                                         &quad_vertex_positions_buffer,
                                         &quad_vertex_indices_buffer);
        ClipVertexStorage {
            vertex_buffer,
            tile_clip_combine_vertex_array,
            tile_clip_copy_vertex_array,
            size,
        }
    }
}

impl<D> Storage for ClipVertexStorage<D> where D: Device {
    fn gpu_bytes_allocated(&self) -> u64 {
        self.size * mem::size_of::<Clip>() as u64
    }
}

*/

// Texture cache

/*
pub(crate) struct TextureCache<D> where D: Device {
    textures: Vec<D::Texture>,
}

impl<D> TextureCache<D> where D: Device {
    pub(crate) fn new() -> TextureCache<D> {
        TextureCache { textures: vec![] }
    }

    pub(crate) fn create_texture(&mut self, device: &mut D, format: TextureFormat, size: Vector2I)
                                 -> D::Texture {
        for index in 0..self.textures.len() {
            if device.texture_size(&self.textures[index]) == size &&
                    device.texture_format(&self.textures[index]) == format {
                return self.textures.remove(index);
            }
        }

        device.create_texture(format, size)
    }

    pub(crate) fn release_texture(&mut self, texture: D::Texture) {
        if self.textures.len() == TEXTURE_CACHE_SIZE {
            self.textures.pop();
        }
        self.textures.insert(0, texture);
    }
}
*/

pub(crate) struct PatternTexturePage {
    pub(crate) framebuffer_id: FramebufferID,
    pub(crate) must_preserve_contents: bool,
}

// Z-buffer

pub(crate) struct ZBuffer<D> where D: Device {
    pub(crate) buffer: Option<D::Buffer>,
    pub(crate) framebuffer: D::Framebuffer,
    pub(crate) tile_size: Vector2I,
}

impl<D> ZBuffer<D> where D: Device {
    fn new(device: &D, renderer_level: RendererLevel, framebuffer_size: Vector2I) -> ZBuffer<D> {
        let tile_size =
            vec2i((framebuffer_size.x() + TILE_WIDTH as i32 - 1) / TILE_WIDTH as i32,
                  (framebuffer_size.y() + TILE_HEIGHT as i32 - 1) / TILE_HEIGHT as i32);

        let buffer = match renderer_level {
            RendererLevel::D3D9 => None,
            RendererLevel::D3D11 => {
                let buffer = device.create_buffer(BufferUploadMode::Dynamic);
                device.allocate_buffer::<u32>(&buffer,
                                              BufferData::Uninitialized(tile_size.area() as usize),
                                              BufferTarget::Storage);
                Some(buffer)
            }
        };

        let texture = device.create_texture(TextureFormat::RGBA8, tile_size);
        device.set_texture_sampling_mode(&texture,
                                         TextureSamplingFlags::NEAREST_MIN |
                                         TextureSamplingFlags::NEAREST_MAG);
        let framebuffer = device.create_framebuffer(texture);
        ZBuffer { buffer, framebuffer, tile_size }
    }
}

/*
impl<D> Storage for ZBuffer<D> where D: Device {
    fn gpu_bytes_allocated(&self) -> u64 {
        let mut size = self.tile_size.area() as u64 * 4;
        if self.buffer.is_some() {
            size += self.tile_size.area() as u64 * 4;
        }
        size
    }
}
*/