// src/backends/cuda.rs  (v3 — native kernel, no cuFFT)
//
// BlitzFFT's own Cooley-Tukey kernel replaces cuFFT.
// The kernel (shaders/blitz_fft.cu) is compiled to PTX by nvcc at build time.
// The PTX bytes are embedded directly into the binary.
//
// Memory management uses the CUDA Runtime API (loaded at runtime via libloading).
// Kernel loading and launching use the CUDA Driver API (libcuda.so / nvcuda.dll).
//
// If nvcc was not found at build time, BlitzCudaKernelAbsent is returned from
// try_init(), and the auto-selector falls through to the next backend.

#![cfg(feature = "cuda")]

use super::{FftBackend, FftFrame};
use anyhow::{anyhow, Result};
use libloading::Library;
use rayon::prelude::*;
use std::{
    ffi::{c_void, CStr},
    sync::Mutex,
};

// ── PTX bytes embedded at compile time ────────────────────────────────────────

#[cfg(blitz_cuda_kernel)]
const BLITZ_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/blitz_fft.ptx"));

// ── Type aliases ──────────────────────────────────────────────────────────────

type CudaError  = i32;
type CUresult   = i32;
type CUmodule   = *mut c_void;
type CUfunction = *mut c_void;
type CudaStream = *mut c_void;

const CUDA_SUCCESS: CudaError = 0;
const CUDA_SUCCESS_CU: CUresult = 0;

const H2D: i32 = 1;
const D2H: i32 = 2;
const PINNED_WRITE_COMBINED: u32 = 0x04;

const SHMEM_MAX_N: usize = 2048;

// ── Runtime API function pointer types ────────────────────────────────────────

type FnCudaMalloc          = unsafe extern "C" fn(*mut *mut c_void, usize) -> CudaError;
type FnCudaFree            = unsafe extern "C" fn(*mut c_void) -> CudaError;
type FnCudaMemcpyAsync     = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32, CudaStream) -> CudaError;
type FnCudaHostAlloc       = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> CudaError;
type FnCudaFreeHost        = unsafe extern "C" fn(*mut c_void) -> CudaError;
type FnCudaStreamCreate    = unsafe extern "C" fn(*mut CudaStream) -> CudaError;
type FnCudaStreamDestroy   = unsafe extern "C" fn(CudaStream) -> CudaError;
type FnCudaStreamSync      = unsafe extern "C" fn(CudaStream) -> CudaError;
type FnCudaDeviceSync      = unsafe extern "C" fn() -> CudaError;

// ── Driver API function pointer types ─────────────────────────────────────────

type FnCuInit              = unsafe extern "C" fn(u32) -> CUresult;
type FnCuModuleLoadData    = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type FnCuModuleGetFunction = unsafe extern "C" fn(*mut CUfunction, CUmodule, *const i8) -> CUresult;
#[allow(clippy::type_complexity)]
type FnCuLaunchKernel      = unsafe extern "C" fn(
    CUfunction,
    u32, u32, u32,       // gridDim
    u32, u32, u32,       // blockDim
    u32,                  // sharedMemBytes
    CudaStream,           // hStream (compatible with cudaStream_t)
    *mut *mut c_void,    // kernelParams
    *mut *mut c_void,    // extra
) -> CUresult;

// ── Function pointer bundles ──────────────────────────────────────────────────

#[allow(dead_code)]
struct RuntimeFns {
    cuda_malloc:        FnCudaMalloc,
    cuda_free:          FnCudaFree,
    cuda_memcpy_async:  FnCudaMemcpyAsync,
    cuda_host_alloc:    FnCudaHostAlloc,
    cuda_free_host:     FnCudaFreeHost,
    cuda_stream_create: FnCudaStreamCreate,
    cuda_stream_destroy:FnCudaStreamDestroy,
    cuda_stream_sync:   FnCudaStreamSync,
    cuda_device_sync:   FnCudaDeviceSync,  // available for manual sync if needed
}

struct DriverFns {
    cu_module_load_data:    FnCuModuleLoadData,
    cu_module_get_function: FnCuModuleGetFunction,
    cu_launch_kernel:       FnCuLaunchKernel,
}

// ── Device buffer pool ────────────────────────────────────────────────────────

struct DevBufs {
    d_in:   *mut c_void,  // real f32 input
    d_out:  *mut c_void,  // complex float2 output [batch × N]
    d_mag:  *mut c_void,  // magnitude output [batch × half1]
    d_win:  *mut c_void,  // Hann window [N]
    d_params: *mut c_void,// params uint32[6]
    h_in:   *mut f32,     // pinned host input
    h_mag:  *mut f32,     // pinned host magnitude
    batch:  usize,
    n:      usize,
}

// ── Kernel handles ────────────────────────────────────────────────────────────

struct Kernels {
    module:    CUmodule,
    shared:    CUfunction,  // blitz_fft_shared
    bit_rev:   CUfunction,  // blitz_bit_rev
    fft_pass:  CUfunction,  // blitz_fft_pass
    magnitude: CUfunction,  // blitz_magnitude
}

// ── Backend ───────────────────────────────────────────────────────────────────

pub struct CudaFftBackend {
    _rt_lib:   Library,
    _drv_lib:  Library,
    rt:        RuntimeFns,
    drv:       DriverFns,
    kernels:   Kernels,
    stream:    CudaStream,
    dev_bufs:  Mutex<Option<DevBufs>>,
}

unsafe impl Send for CudaFftBackend {}
unsafe impl Sync for CudaFftBackend {}

impl CudaFftBackend {
    pub fn try_init() -> Option<Self> {
        // Native kernel requires PTX compiled at build time.
        #[cfg(not(blitz_cuda_kernel))]
        {
            eprintln!(
                "[BlitzFFT] CUDA native kernel: nvcc not found at build time. \
                 Install the CUDA Toolkit and rebuild to enable the CUDA backend."
            );
            return None;
        }

        #[cfg(blitz_cuda_kernel)]
        Self::try_init_with_ptx()
    }

    #[cfg(blitz_cuda_kernel)]
    fn try_init_with_ptx() -> Option<Self> {
        // Load runtime library.
        let rt_names = ["libcudart.so.12", "libcudart.so.11", "cudart64_12.dll", "cudart64_110.dll"];
        let rt_lib = rt_names.iter().find_map(|n| unsafe { Library::new(n) }.ok())?;

        // Load driver library.
        let drv_names = ["libcuda.so.1", "libcuda.so", "nvcuda.dll"];
        let drv_lib = drv_names.iter().find_map(|n| unsafe { Library::new(n) }.ok())?;

        macro_rules! rt_sym {
            ($name:literal, $ty:ty) => {
                *unsafe { rt_lib.get::<$ty>($name).ok()? }
            };
        }
        macro_rules! drv_sym {
            ($name:literal, $ty:ty) => {
                *unsafe { drv_lib.get::<$ty>($name).ok()? }
            };
        }

        let rt = RuntimeFns {
            cuda_malloc:        rt_sym!(b"cudaMalloc\0",              FnCudaMalloc),
            cuda_free:          rt_sym!(b"cudaFree\0",                FnCudaFree),
            cuda_memcpy_async:  rt_sym!(b"cudaMemcpyAsync\0",         FnCudaMemcpyAsync),
            cuda_host_alloc:    rt_sym!(b"cudaHostAlloc\0",           FnCudaHostAlloc),
            cuda_free_host:     rt_sym!(b"cudaFreeHost\0",            FnCudaFreeHost),
            cuda_stream_create: rt_sym!(b"cudaStreamCreate\0",        FnCudaStreamCreate),
            cuda_stream_destroy:rt_sym!(b"cudaStreamDestroy\0",       FnCudaStreamDestroy),
            cuda_stream_sync:   rt_sym!(b"cudaStreamSynchronize\0",   FnCudaStreamSync),
            cuda_device_sync:   rt_sym!(b"cudaDeviceSynchronize\0",   FnCudaDeviceSync),
        };

        let cu_init: FnCuInit = drv_sym!(b"cuInit\0", FnCuInit);
        unsafe { cu_init(0); } // Initialize driver API.

        let drv = DriverFns {
            cu_module_load_data:    drv_sym!(b"cuModuleLoadData\0",    FnCuModuleLoadData),
            cu_module_get_function: drv_sym!(b"cuModuleGetFunction\0", FnCuModuleGetFunction),
            cu_launch_kernel:       drv_sym!(b"cuLaunchKernel\0",      FnCuLaunchKernel),
        };

        // Load PTX module.
        let mut module: CUmodule = std::ptr::null_mut();
        let cu_result = unsafe {
            (drv.cu_module_load_data)(&mut module, BLITZ_PTX.as_ptr() as *const c_void)
        };
        if cu_result != CUDA_SUCCESS_CU {
            eprintln!("[BlitzFFT] cuModuleLoadData failed: {cu_result}");
            return None;
        }

        // Retrieve kernel functions.
        let get_fn = |name: &CStr| -> Option<CUfunction> {
            let mut func: CUfunction = std::ptr::null_mut();
            let r = unsafe { (drv.cu_module_get_function)(&mut func, module, name.as_ptr()) };
            if r != CUDA_SUCCESS_CU { None } else { Some(func) }
        };

        let shared    = get_fn(c"blitz_fft_shared")?;
        let bit_rev   = get_fn(c"blitz_bit_rev")?;
        let fft_pass  = get_fn(c"blitz_fft_pass")?;
        let magnitude = get_fn(c"blitz_magnitude")?;

        let kernels = Kernels { module, shared, bit_rev, fft_pass, magnitude };

        // Create a persistent stream.
        let mut stream: CudaStream = std::ptr::null_mut();
        unsafe { (rt.cuda_stream_create)(&mut stream); }

        Some(Self {
            _rt_lib: rt_lib,
            _drv_lib: drv_lib,
            rt,
            drv,
            kernels,
            stream,
            dev_bufs: Mutex::new(None),
        })
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    unsafe fn ensure_bufs(&self, batch: usize, n: usize) -> Result<()> {
        let mut guard = self.dev_bufs.lock().unwrap();
        if let Some(ref b) = *guard {
            if b.batch >= batch && b.n == n { return Ok(()); }
            // Free old.
            (self.rt.cuda_free)(b.d_in);
            (self.rt.cuda_free)(b.d_out);
            (self.rt.cuda_free)(b.d_mag);
            (self.rt.cuda_free)(b.d_win);
            (self.rt.cuda_free)(b.d_params);
            (self.rt.cuda_free_host)(b.h_in as _);
            (self.rt.cuda_free_host)(b.h_mag as _);
        }

        let half1    = n / 2 + 1;
        let in_bytes = batch * n * 4;
        let out_bytes= batch * n * 8;
        let mag_bytes= batch * half1 * 4;
        let win_bytes= n * 4;
        let par_bytes= 6 * 4; // 6 × uint32

        let alloc = |bytes| -> Result<*mut c_void> {
            let mut p: *mut c_void = std::ptr::null_mut();
            let r = (self.rt.cuda_malloc)(&mut p, bytes);
            if r != CUDA_SUCCESS { Err(anyhow!("cudaMalloc({bytes}) failed: {r}")) } else { Ok(p) }
        };

        let d_in     = alloc(in_bytes)?;
        let d_out    = alloc(out_bytes)?;
        let d_mag    = alloc(mag_bytes)?;
        let d_win    = alloc(win_bytes)?;
        let d_params = alloc(par_bytes)?;

        let mut h_in:  *mut c_void = std::ptr::null_mut();
        let mut h_mag: *mut c_void = std::ptr::null_mut();
        if (self.rt.cuda_host_alloc)(&mut h_in,  in_bytes,  PINNED_WRITE_COMBINED) != CUDA_SUCCESS
        || (self.rt.cuda_host_alloc)(&mut h_mag, mag_bytes, 0) != CUDA_SUCCESS {
            return Err(anyhow!("cudaHostAlloc failed"));
        }

        *guard = Some(DevBufs {
            d_in, d_out, d_mag, d_win, d_params,
            h_in: h_in as *mut f32, h_mag: h_mag as *mut f32,
            batch, n,
        });
        Ok(())
    }

    unsafe fn write_params(d_params: *mut c_void, stage: u32, n: u32, batch: u32,
                            use_win: u32, stream: CudaStream, rt: &RuntimeFns) {
        let log2n = (n as f32).log2() as u32;
        let half1 = n / 2 + 1;
        let arr: [u32; 6] = [stage, n, batch, use_win, log2n, half1];
        (rt.cuda_memcpy_async)(
            d_params,
            arr.as_ptr() as *const c_void,
            24,
            H2D,
            stream,
        );
    }
}

impl Drop for CudaFftBackend {
    fn drop(&mut self) {
        unsafe {
            (self.rt.cuda_stream_destroy)(self.stream);
            if let Ok(guard) = self.dev_bufs.lock() {
                if let Some(ref b) = *guard {
                    (self.rt.cuda_free)(b.d_in);
                    (self.rt.cuda_free)(b.d_out);
                    (self.rt.cuda_free)(b.d_mag);
                    (self.rt.cuda_free)(b.d_win);
                    (self.rt.cuda_free)(b.d_params);
                    (self.rt.cuda_free_host)(b.h_in as _);
                    (self.rt.cuda_free_host)(b.h_mag as _);
                }
            }
        }
    }
}

impl FftBackend for CudaFftBackend {
    fn name(&self) -> &str {
        "BlitzFFT native CUDA kernel (v3 — no cuFFT)"
    }

    fn compute_batch(&self, frames: &[&[f32]], fft_size: usize) -> Result<Vec<FftFrame>> {
        let n     = fft_size;
        let batch = frames.len();
        let half1 = n / 2 + 1;

        unsafe {
            self.ensure_bufs(batch, n)?;

            let guard = self.dev_bufs.lock().unwrap();
            let b = guard.as_ref().unwrap();
            let stream = self.stream;

            // ── 1. Fill pinned input buffer ───────────────────────────────
            for (i, frame) in frames.iter().enumerate() {
                let dst = b.h_in.add(i * n);
                let len = frame.len().min(n);
                std::ptr::copy_nonoverlapping(frame.as_ptr(), dst, len);
                if len < n { std::ptr::write_bytes(dst.add(len), 0, n - len); }
            }

            // ── 2. H→D transfer ───────────────────────────────────────────
            (self.rt.cuda_memcpy_async)(b.d_in, b.h_in as _, batch * n * 4, H2D, stream);

            if n <= SHMEM_MAX_N {
                // ── 3a. Single-pass shared-memory FFT ─────────────────────
                // Grid: (batch, 1, 1), Block: (N/2, 1, 1)
                // Shared mem: N × 8 bytes

                Self::write_params(b.d_params, 0, n as u32, batch as u32, 0, stream, &self.rt);

                // Kernel params as device pointers / scalars — passed via pointer array.
                let mut p0: *mut c_void = b.d_in;
                // Window pointer: pass null (use_window=0 for now; framed path windows on CPU)
                let mut p1: *mut c_void = std::ptr::null_mut();
                let mut p2: *mut c_void = b.d_out;
                let mut p3: *mut c_void = b.d_mag;
                let mut p4: *mut c_void = b.d_params;
                let mut kp: [*mut c_void; 5] = [
                    &mut p0 as *mut *mut c_void as *mut c_void,
                    &mut p1 as *mut *mut c_void as *mut c_void,
                    &mut p2 as *mut *mut c_void as *mut c_void,
                    &mut p3 as *mut *mut c_void as *mut c_void,
                    &mut p4 as *mut *mut c_void as *mut c_void,
                ];

                let shmem_bytes = (n * 8) as u32;
                let r = (self.drv.cu_launch_kernel)(
                    self.kernels.shared,
                    batch as u32, 1, 1,       // grid
                    (n / 2) as u32, 1, 1,     // block (N/2 threads)
                    shmem_bytes,
                    stream,
                    kp.as_mut_ptr(),
                    std::ptr::null_mut(),
                );
                if r != CUDA_SUCCESS_CU {
                    return Err(anyhow!("blitz_fft_shared launch failed: {r}"));
                }
            } else {
                // ── 3b. Multi-pass global-memory FFT ──────────────────────
                let log2n = (n as f64).log2() as u32;
                let threads_per_block: u32 = 512;

                // Step A: bit-reversal
                {
                    Self::write_params(b.d_params, 0, n as u32, batch as u32, 0, stream, &self.rt);
                    let grid_x = ((n as u32) + threads_per_block - 1) / threads_per_block;
                    let mut p0: *mut c_void = b.d_out;
                    let mut p1: *mut c_void = b.d_in;
                    let mut p2: *mut c_void = std::ptr::null_mut(); // no window
                    let mut p3: *mut c_void = b.d_params;
                    let mut kp: [*mut c_void; 4] = [
                        &mut p0 as *mut *mut c_void as *mut c_void,
                        &mut p1 as *mut *mut c_void as *mut c_void,
                        &mut p2 as *mut *mut c_void as *mut c_void,
                        &mut p3 as *mut *mut c_void as *mut c_void,
                    ];
                    (self.drv.cu_launch_kernel)(
                        self.kernels.bit_rev,
                        grid_x, batch as u32, 1,
                        threads_per_block, 1, 1,
                        0, stream, kp.as_mut_ptr(), std::ptr::null_mut(),
                    );
                }

                // Step B: butterfly stages
                for stage in 0..log2n {
                    Self::write_params(b.d_params, stage, n as u32, batch as u32, 0, stream, &self.rt);
                    let half = (n / 2) as u32;
                    let grid_x = (half + threads_per_block - 1) / threads_per_block;
                    let mut p0: *mut c_void = b.d_out;
                    let mut p1: *mut c_void = b.d_params;
                    let mut kp: [*mut c_void; 2] = [
                        &mut p0 as *mut *mut c_void as *mut c_void,
                        &mut p1 as *mut *mut c_void as *mut c_void,
                    ];
                    (self.drv.cu_launch_kernel)(
                        self.kernels.fft_pass,
                        grid_x, batch as u32, 1,
                        threads_per_block, 1, 1,
                        0, stream, kp.as_mut_ptr(), std::ptr::null_mut(),
                    );
                }

                // Step C: magnitude
                {
                    Self::write_params(b.d_params, 0, n as u32, batch as u32, 0, stream, &self.rt);
                    let grid_x = (half1 as u32 + threads_per_block - 1) / threads_per_block;
                    let mut p0: *mut c_void = b.d_out;
                    let mut p1: *mut c_void = b.d_mag;
                    let mut p2: *mut c_void = b.d_params;
                    let mut kp: [*mut c_void; 3] = [
                        &mut p0 as *mut *mut c_void as *mut c_void,
                        &mut p1 as *mut *mut c_void as *mut c_void,
                        &mut p2 as *mut *mut c_void as *mut c_void,
                    ];
                    (self.drv.cu_launch_kernel)(
                        self.kernels.magnitude,
                        grid_x, batch as u32, 1,
                        threads_per_block, 1, 1,
                        0, stream, kp.as_mut_ptr(), std::ptr::null_mut(),
                    );
                }
            }

            // ── 4. D→H transfer (magnitude only) ─────────────────────────
            let mag_bytes = batch * half1 * 4;
            (self.rt.cuda_memcpy_async)(b.h_mag as _, b.d_mag, mag_bytes, D2H, stream);

            // ── 5. Sync ───────────────────────────────────────────────────
            (self.rt.cuda_stream_sync)(stream);

            // ── 6. Collect results ────────────────────────────────────────
            let mag_ptr = b.h_mag;
            let results: Vec<FftFrame> = (0..batch)
                .into_par_iter()
                .map(|i| {
                    let base = i * half1;
                    let magnitude = (0..half1)
                        .map(|k| unsafe { *mag_ptr.add(base + k) })
                        .collect();
                    FftFrame { frame_index: i, magnitude }
                })
                .collect();

            Ok(results)
        }
    }
}
