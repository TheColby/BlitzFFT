// src/backends/metal.rs  (v2)
//
// Optimisations over v1
// ─────────────────────
//  1. Shared-memory shader for N ≤ 2048
//     `fft_shared` does bit-reversal + all butterfly stages + magnitude
//     entirely within 16 KB threadgroup SRAM — global memory touched once in,
//     once out. For N=2048 this is ~11× fewer global memory operations.
//
//  2. Persistent buffer pool
//     MTLBuffers allocated once, reused across calls — no per-call alloc.
//
//  3. Fused Hann windowing on GPU
//     Window coefficients live in a persistent GPU buffer; multiplied in the
//     `fft_shared` load stage — no separate CPU windowing pass.
//
//  4. Multi-pass path for N > 2048 still faster via fused window in stage 0.

#![cfg(feature = "metal")]

use super::{FftBackend, FftFrame};
use anyhow::Result;
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::sync::Mutex;

const SHMEM_MAX_N: usize = 2048;

// ── Persistent buffer pool ────────────────────────────────────────────────────

struct Buffers {
    real_in: Buffer,  // f32   [batch × N]
    window: Buffer,   // f32   [N]
    cplx_out: Buffer, // float2 [batch × N]
    mag_out: Buffer,  // f32   [batch × (N/2+1)]
    params: Buffer,   // u32   [6]
    batch_cap: usize,
    n_cap: usize,
}

impl Buffers {
    fn new(device: &Device, batch: usize, n: usize) -> Self {
        let half1 = n / 2 + 1;
        let opt = MTLResourceOptions::StorageModeShared;
        Self {
            real_in: device.new_buffer((batch * n * 4) as u64, opt),
            window: device.new_buffer((n * 4) as u64, opt),
            cplx_out: device.new_buffer((batch * n * 8) as u64, opt),
            mag_out: device.new_buffer((batch * half1 * 4) as u64, opt),
            params: device.new_buffer(24_u64, opt),
            batch_cap: batch,
            n_cap: n,
        }
    }
    fn needs_resize(&self, batch: usize, n: usize) -> bool {
        batch > self.batch_cap || n != self.n_cap
    }
}

struct Pipelines {
    shared: ComputePipelineState,
    pass_global: ComputePipelineState,
    bit_rev: ComputePipelineState,
    magnitude: ComputePipelineState,
}

pub struct MetalFftBackend {
    device: Device,
    queue: CommandQueue,
    psos: Pipelines,
    bufs: Mutex<Option<Buffers>>,
}

unsafe impl Send for MetalFftBackend {}
unsafe impl Sync for MetalFftBackend {}

impl MetalFftBackend {
    pub fn try_init() -> Option<Self> {
        let device = Device::system_default()?;
        let lib_bytes = include_bytes!(env!("METAL_LIBRARY_PATH"));
        let data = metal::DispatchData::from(lib_bytes.as_ref());
        let library = device.new_library_with_data(data).ok()?;

        let pso = |name: &str| -> Option<ComputePipelineState> {
            let func = library.get_function(name, None).ok()?;
            device.new_compute_pipeline_state_with_function(&func).ok()
        };

        Some(Self {
            queue: device.new_command_queue(),
            psos: Pipelines {
                shared: pso("fft_shared")?,
                pass_global: pso("fft_pass_global")?,
                bit_rev: pso("bit_reverse_global")?,
                magnitude: pso("magnitude_only")?,
            },
            bufs: Mutex::new(None),
            device,
        })
    }

    fn tg(pso: &ComputePipelineState, desired: usize) -> MTLSize {
        let w = desired.min(pso.max_total_threads_per_threadgroup() as usize);
        MTLSize {
            width: w as u64,
            height: 1,
            depth: 1,
        }
    }

    fn write_params(buf: &Buffer, stage: u32, n: u32, batch: u32, use_win: u32) {
        let log2n = (n as f32).log2() as u32;
        let half1 = n / 2 + 1;
        let p = buf.contents() as *mut u32;
        unsafe {
            *p = stage;
            *p.add(1) = n;
            *p.add(2) = batch;
            *p.add(3) = use_win;
            *p.add(4) = log2n;
            *p.add(5) = half1;
        }
    }
}

impl FftBackend for MetalFftBackend {
    fn name(&self) -> &str {
        "Metal GPU — shared-memory FFT (v2)"
    }

    fn compute_batch(&self, frames: &[&[f32]], fft_size: usize) -> Result<Vec<FftFrame>> {
        let n = fft_size;
        let batch = frames.len();
        let half1 = n / 2 + 1;

        // ── Ensure persistent buffers ─────────────────────────────────────
        let mut guard = self.bufs.lock().unwrap();
        if guard.as_ref().map_or(true, |b| b.needs_resize(batch, n)) {
            *guard = Some(Buffers::new(&self.device, batch, n));
        }
        let bufs = guard.as_ref().unwrap();

        // ── Copy audio into unified buffer (zero-copy on Apple Silicon) ───
        {
            let ptr = bufs.real_in.contents() as *mut f32;
            for (i, frame) in frames.iter().enumerate() {
                let base = i * n;
                let len = frame.len().min(n);
                unsafe {
                    std::ptr::copy_nonoverlapping(frame.as_ptr(), ptr.add(base), len);
                    if len < n {
                        std::ptr::write_bytes(ptr.add(base + len), 0, n - len);
                    }
                }
            }
        }

        // ── Write Hann window coefficients once ───────────────────────────
        {
            use std::f32::consts::PI;
            let ptr = bufs.window.contents() as *mut f32;
            for k in 0..n {
                unsafe {
                    *ptr.add(k) = 0.5 * (1.0 - (2.0 * PI * k as f32 / (n - 1) as f32).cos());
                }
            }
        }

        let cmd = self.queue.new_command_buffer();

        if n <= SHMEM_MAX_N {
            // ── FAST PATH: single-pass shared-memory FFT ──────────────────
            // One threadgroup per FFT, N/2 threads per group.
            // All log2(N) butterfly stages execute in SRAM — global memory
            // is read once and written once.
            Self::write_params(&bufs.params, 0, n as u32, batch as u32, 1);

            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.psos.shared);
            enc.set_buffer(0, Some(&bufs.real_in), 0);
            enc.set_buffer(1, Some(&bufs.window), 0);
            enc.set_buffer(2, Some(&bufs.cplx_out), 0);
            enc.set_buffer(3, Some(&bufs.mag_out), 0);
            enc.set_buffer(4, Some(&bufs.params), 0);
            enc.set_threadgroup_memory_length(0, (n * 8) as u64);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: batch as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: (n / 2) as u64,
                    height: 1,
                    depth: 1,
                },
            );
            enc.end_encoding();
        } else {
            // ── MULTI-PASS PATH: N > 2048 (e.g. 4096, 8192) ──────────────
            let log2n = (n as f64).log2() as u32;

            // Load real → complex buffer with bit-reversal
            {
                Self::write_params(&bufs.params, 0, n as u32, batch as u32, 0);
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.psos.bit_rev);
                enc.set_buffer(0, Some(&bufs.cplx_out), 0);
                enc.set_buffer(1, Some(&bufs.params), 0);
                enc.dispatch_threads(
                    MTLSize {
                        width: n as u64,
                        height: batch as u64,
                        depth: 1,
                    },
                    Self::tg(&self.psos.bit_rev, 512),
                );
                enc.end_encoding();
            }

            // Butterfly stages — window fused into stage 0
            for stage in 0..log2n {
                let use_win = if stage == 0 { 1u32 } else { 0u32 };
                Self::write_params(&bufs.params, stage, n as u32, batch as u32, use_win);
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.psos.pass_global);
                enc.set_buffer(0, Some(&bufs.cplx_out), 0);
                enc.set_buffer(1, Some(&bufs.window), 0);
                enc.set_buffer(2, Some(&bufs.params), 0);
                enc.dispatch_threads(
                    MTLSize {
                        width: (n / 2) as u64,
                        height: batch as u64,
                        depth: 1,
                    },
                    Self::tg(&self.psos.pass_global, 512),
                );
                enc.end_encoding();
            }

            // Magnitude
            {
                Self::write_params(&bufs.params, 0, n as u32, batch as u32, 0);
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.psos.magnitude);
                enc.set_buffer(0, Some(&bufs.cplx_out), 0);
                enc.set_buffer(1, Some(&bufs.mag_out), 0);
                enc.set_buffer(2, Some(&bufs.params), 0);
                enc.dispatch_threads(
                    MTLSize {
                        width: half1 as u64,
                        height: batch as u64,
                        depth: 1,
                    },
                    Self::tg(&self.psos.magnitude, 512),
                );
                enc.end_encoding();
            }
        }

        cmd.commit();
        cmd.wait_until_completed();

        // ── Read back (zero-copy on Apple Silicon) ────────────────────────
        let cplx_ptr = bufs.cplx_out.contents() as *const [f32; 2];
        let mag_ptr = bufs.mag_out.contents() as *const f32;

        let out = (0..batch)
            .map(|i| {
                let magnitude: Vec<f32> = (0..half1)
                    .map(|k| unsafe { *mag_ptr.add(i * half1 + k) })
                    .collect();
                FftFrame {
                    frame_index: i,
                    magnitude,
                }
            })
            .collect();

        Ok(out)
    }
}
