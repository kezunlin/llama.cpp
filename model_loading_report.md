# Model Loading Phase in llama.cpp with GPU Offloading

This report details the model loading phase in the `llama.cpp` project, with a particular focus on how GPU offloading is handled. The analysis is based on `common/common.cpp`, `src/llama.cpp`, and `src/llama-model-loader.cpp`.

## 1. Parameter Translation

The process begins with user-specified parameters, often parsed from command-line arguments or set in code. These high-level parameters are translated into structures understood by the core library.

*   **`common_init_from_params()` (`common/common.cpp`):** This function is a high-level entry point typically used in examples like `main.cpp`. It orchestrates model and context initialization. For model loading, it calls `common_model_params_to_llama()`.
    ```c++
    // In common_init_from_params()
    auto mparams = common_model_params_to_llama(params); // params is common_params
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    ```

*   **`common_model_params_to_llama()` (`common/common.cpp`):** This function takes a `common_params` structure (which holds user-friendly settings) and populates a `llama_model_params` structure. This is the crucial translation step for model loading parameters.
    ```c++
    // In common_model_params_to_llama()
    auto mparams = llama_model_default_params(); // Get defaults

    // GPU offloading related parameters:
    if (params.n_gpu_layers != -1) { // params is common_params
        mparams.n_gpu_layers = params.n_gpu_layers;
    }
    mparams.main_gpu     = params.main_gpu;
    mparams.split_mode   = params.split_mode; // How to split (none, layer, row)
    mparams.tensor_split = params.tensor_split; // float array for splitting ratios across GPUs

    // Memory mapping options:
    mparams.use_mmap     = params.use_mmap;
    mparams.use_mlock    = params.use_mlock;

    // Other relevant parameters:
    mparams.check_tensors = params.check_tensors;
    if (!params.devices.empty()) { // User can explicitly provide a list of devices
        mparams.devices = params.devices.data();
    }
    mparams.kv_overrides = params.kv_overrides.empty() ? NULL : params.kv_overrides.data();
    // ... other parameters ...
    return mparams;
    ```

## 2. Initial Model Loading Steps (`src/llama.cpp`)

With the `llama_model_params` populated, the actual model loading begins.

*   **`llama_model_load_from_file()` (public API):** This function is the typical entry point for loading a model. It calls the internal implementation `llama_model_load_from_file_impl()`.
    ```c++
    // Public API
    struct llama_model * llama_model_load_from_file(
            const char * path_model,
            struct llama_model_params params) {
        std::vector<std::string> splits = {};
        return llama_model_load_from_file_impl(path_model, splits, params);
    }
    ```

*   **`llama_model_load_from_file_impl()` (`src/llama.cpp`):**
    *   It first initializes `ggml_time_init()` and checks if any backends are loaded if not in `vocab_only` mode.
    *   It creates a `llama_model` object: `llama_model * model = new llama_model(params);`.
    *   **Device List Determination:**
        *   If `params.devices` (the `ggml_backend_dev_t *` array in `llama_model_params`) is explicitly provided, those devices are used.
        *   Otherwise, it iterates through all registered backends using `ggml_backend_dev_count()` and `ggml_backend_dev_get(i)`. It filters for `GGML_BACKEND_DEVICE_TYPE_GPU` and `GGML_BACKEND_DEVICE_TYPE_ACCEL`, adding them to `model->devices`. RPC backends are typically added first.
        *   **Single GPU Mode:** If `params.split_mode == LLAMA_SPLIT_MODE_NONE`, the code ensures `params.main_gpu` is valid and then clears `model->devices`, adding only the `ggml_backend_dev_t` corresponding to `params.main_gpu`.
            ```c++
            // In llama_model_load_from_file_impl()
            if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
                if (params.main_gpu < 0 || params.main_gpu >= (int)model->devices.size()) { /* error */ }
                ggml_backend_dev_t main_gpu_dev = model->devices[params.main_gpu];
                model->devices.clear();
                model->devices.push_back(main_gpu_dev);
            }
            ```
    *   **`llama_model_load()` Call:** This internal function is then called, which orchestrates the actual loading process using `llama_model_loader`.
        ```c++
        // In llama_model_load_from_file_impl()
        const int status = llama_model_load(path_model, splits, *model, params);
        ```
        Inside `llama_model_load()`:
        ```c++
        // In llama_model_load()
        try {
            llama_model_loader ml(fname, splits, params.use_mmap, params.check_tensors, params.kv_overrides, params.tensor_buft_overrides);
            // ... load architecture, hparams, vocab ...
            if (params.vocab_only) { /* skip tensors */ return 0; }
            if (!model.load_tensors(ml)) { /* handle cancellation */ }
        } catch (const std::exception & err) { /* handle error */ }
        ```
    *   The `llama_model_loader` instance (`ml`) is created here, passing `params.use_mmap`, `params.check_tensors`, and overrides.

## 3. GGUF Parsing and Tensor Metadata (`src/llama-model-loader.cpp` - Constructor)

The `llama_model_loader` class is responsible for handling the GGUF file(s) and extracting metadata.

*   **`llama_model_loader::llama_model_loader()` (Constructor):**
    *   **Main GGUF Parsing:** It initializes a `gguf_context` from the main model file (`fname`) using `gguf_init_from_file()`. The `no_alloc` parameter in `gguf_init_params` is set to `true`, meaning it only reads metadata, not tensor data.
        ```c++
        // In llama_model_loader constructor
        meta.reset(gguf_init_from_file(fname.c_str(), params)); // params has .no_alloc = true
        files.emplace_back(new llama_file(fname.c_str(), "rb"));
        contexts.emplace_back(ctx); // ctx from gguf_init_params
        ```
    *   **`weights_map` Population (Main File):** It iterates through all tensors in the main GGUF context using `ggml_get_first_tensor()` and `ggml_get_next_tensor()`. For each tensor, it creates a `llama_tensor_weight` entry in `weights_map`. This entry stores:
        *   A pointer to the `llama_file` object.
        *   The split index (0 for the main file).
        *   The `gguf_context` of the file.
        *   The `ggml_tensor*` containing metadata (shape, type, name) but no actual data yet.
        *   The offset of the tensor's data within the GGUF file (`w.offs`, derived from `ggml_get_data_offset(cur)` during GGUF parsing by `gguf_init_from_file`).
        ```c++
        // In llama_model_loader constructor, for the main file
        for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
            // ...
            weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, meta.get(), cur));
        }
        ```
    *   **Split GGUF Files:** If the model is split (indicated by `n_split > 1` in GGUF metadata), it iterates through the split files. For each split:
        *   It opens the split GGUF file and initializes another `gguf_context`.
        *   It populates `weights_map` with tensor metadata from that split, similar to the main file, but with the corresponding split index.
    *   **`kv_overrides`:** The `param_overrides_p` (a `const llama_model_kv_override *`) passed to the constructor is used to populate the `kv_overrides` member (an `std::map<std::string, llama_model_kv_override>`). These overrides allow changing model metadata values (like RoPE scaling, GQA, etc.) at load time. The `get_key()` methods in `llama_model_loader` will consult these overrides.

## 4. Memory Mapping (`src/llama-model-loader.cpp` - `init_mappings()`)

If `use_mmap` is enabled, the model files are memory-mapped for direct access.

*   **`llama_model_loader::init_mappings()`:**
    *   This function is called after the constructor.
    *   If `this->use_mmap` (set from `params.use_mmap`) is true:
        *   It iterates through the `files` (vector of `llama_file` pointers).
        *   For each file, it creates a `llama_mmap` object: `std::unique_ptr<llama_mmap> mapping = std::make_unique<llama_mmap>(file.get(), prefetch ? -1 : 0, is_numa);`. This maps the entire file into memory.
        *   These `llama_mmap` objects are stored in the `mappings` vector.
    *   **`mlock`:** If `mlock_mmaps` (a pointer to `llama_mlocks`, which is a `std::vector<llama_mlock_ptr>`) is provided, it initializes an `llama_mlock` object for each mapping and calls `mlock_mmap->init(mapping->addr())`. This attempts to lock the memory-mapped regions into RAM, preventing them from being paged out.
        ```c++
        // In init_mappings()
        if (mlock_mmaps) {
            std::unique_ptr<llama_mlock> mlock_mmap(new llama_mlock());
            mlock_mmap->init(mapping->addr()); // Lock the mapped region
            mlock_mmaps->emplace_back(std::move(mlock_mmap));
        }
        ```

## 5. Tensor Allocation Strategy (Conceptual)

While `llama-model-loader.cpp` prepares all tensor metadata and file mappings, the actual allocation of `ggml_tensor` instances into the model's main `ggml_context` and their assignment to specific backend buffers (CPU or GPU) happens in `llama_model::load_tensors` (defined in `src/llama-model.cpp`, called by `llama_model_load` in `src/llama.cpp`).

This allocation strategy uses:

*   **`params.n_gpu_layers` (via `model.hparams.n_gpu_layers`):** This is the primary control for offloading. `llama_model::load_tensors` iterates through model layers and tensors. If a tensor belongs to a layer index less than `n_gpu_layers`, it's a candidate for GPU offloading. Specific tensor names (like "output.weight" or "tok_embeddings.weight") are also often checked for offloading.
*   **`model.devices` (populated based on `params.main_gpu`, `params.tensor_split`):** This vector determines which GPU(s) are available for offloading.
    *   In single-GPU mode (`params.split_mode == LLAMA_SPLIT_MODE_NONE`), `model.devices` contains only one device (`params.main_gpu`). All offloaded tensors go to this GPU.
    *   In multi-GPU mode, `model.devices` contains multiple GPU backends.
*   **`params.tensor_split`:** This array of floats dictates how tensor data should be distributed across the available `model.devices`. For example, a tensor might have its rows split, with a certain percentage of rows going to each GPU. This is handled by `ggml_backend_alloc_split_buffer` or similar mechanisms when creating backend buffers for tensors. `ggml_backend_sched_split_init` and related functions help create schedules for computation on split tensors.
*   **`params.split_mode` (`LLAMA_SPLIT_MODE_LAYER`, `LLAMA_SPLIT_MODE_ROW`):**
    *   `LLAMA_SPLIT_MODE_LAYER`: Entire layers are assigned to different GPUs. The `tensor_split` array might define proportions of layers per GPU.
    *   `LLAMA_SPLIT_MODE_ROW`: Individual tensors (especially large ones like embedding layers or linear layer weights) can have their rows split across GPUs according to `tensor_split`.
*   The result of this stage is that `ggml_tensor` objects are created within the model's `ggml_context` (`model.ctx_w`). Their `ggml_tensor->buffer` field will point to a `ggml_backend_buffer_t` representing memory on the target device (CPU or a specific GPU). However, `ggml_tensor->data` might still be `nullptr` or point to a temporary allocation. The actual data loading comes next.

## 6. Tensor Data Population (`src/llama-model-loader.cpp` - `load_all_data()`)

`llama_model_loader::load_all_data()` is responsible for reading the tensor data from files (or mmap) and placing it into the allocated `ggml_tensor` locations, potentially copying to GPU. This function is called by `llama_model::load_tensors_data`.

*   It iterates through all `ggml_tensor`s in the provided `ggml_context *ctx` (which is `model.ctx_w`).
*   For each tensor `cur`:
    *   It retrieves the corresponding `llama_tensor_weight w = require_weight(ggml_get_name(cur))`.

    *   **Mmap Case (`use_mmap == true`):**
        *   The memory-mapped address for the tensor data is `(uint8_t *)mapping->addr() + w.offs`.
        *   If `cur->data == nullptr` (meaning the tensor's data pointer hasn't been set yet) AND a backend buffer `buf_mmap` exists for this tensor's file split index (`w.idx`):
            *   `ggml_backend_tensor_alloc(buf_mmap, cur, data)` is called. This function from `ggml-backend.h` associates the `ggml_tensor` `cur` with the backend buffer `buf_mmap` and sets `cur->data` to point to the `data` offset within the mmapped region. The backend is now aware that this tensor's data resides in this mmapped host memory.
        *   Else (if `cur->data` is already set, e.g., to a GPU buffer, or no specific `buf_mmap`):
            *   `ggml_backend_tensor_set(cur, data, 0, n_size)` is called. This copies the `n_size` bytes from the mmapped `data` location to the memory location already pointed to by `cur->data` (which could be a GPU buffer).
        *   If `lmlocks` is enabled, `lmlock->grow_to()` ensures the mlocked region covers the tensor.

    *   **Non-Mmap Case (`use_mmap == false`):**
        *   **Host Buffer:** If `ggml_backend_buffer_is_host(cur->buffer)` is true (tensor is on CPU):
            *   `file->seek(w.offs, SEEK_SET);`
            *   `file->read_raw(cur->data, n_size);` The data is read directly into the tensor's memory.
        *   **Non-Host (GPU) Buffer:**
            *   **Synchronous Copy (default if async conditions not met):**
                *   A temporary host buffer `read_buf` is allocated.
                *   `file->seek(w.offs, SEEK_SET);`
                *   `file->read_raw(read_buf.data(), n_size);` Data is read into `read_buf`.
                *   `ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);` This copies the data from the host `read_buf` to the GPU memory associated with `cur`.
            *   **Asynchronous Copy (if `upload_backend` is configured):**
                *   An `upload_backend` is set up if the target device supports async operations, host-pinned buffers, and events (e.g., CUDA backends). Pinned host buffers (`host_buffers`, `host_ptrs`) and events (`events`) are created.
                *   Data is read from the file into a pinned host buffer chunk by chunk: `file->read_raw(host_ptrs[buffer_idx], read_iteration);`.
                *   `ggml_backend_tensor_set_async(upload_backend, cur, host_ptrs[buffer_idx], bytes_read, read_iteration);` This initiates an asynchronous copy from the pinned host buffer to the target GPU tensor `cur`.
                *   `ggml_backend_event_record(events[buffer_idx], upload_backend);` records an event for this transfer.
                *   `ggml_backend_event_synchronize(events[buffer_idx]);` is used before reusing a pinned buffer to ensure the previous async transfer using it has completed.
    *   Progress is reported via `progress_callback`.
    *   If `check_tensors` is true, `ggml_validate_row_data()` is called (potentially asynchronously) to validate tensor data.

## 7. Backend Initialization Prerequisite

For any GPU offloading to work, the necessary backends must be initialized beforehand.

*   **`llama_backend_init()`:** This function (usually called once at program start, e.g., in `common_init()` or at the beginning of `llama_model_load_from_file_impl()`) initializes basic backend infrastructure and GGML time.
*   **Specific Backend Initializers:** Functions like `ggml_backend_cuda_init()` (or equivalents for other GPU types like Metal, SYCL) must be called to register the GPU backends with GGML. These are often called automatically if using `ggml_backend_load_all_backends()` or when a specific backend library (like `ggml-cuda.so`) is loaded.
*   `llama_model_load_from_file_impl()` checks `if (!params.vocab_only && ggml_backend_reg_count() == 0)` and logs an error if no backends are loaded, as tensors cannot be processed without them.

This comprehensive process allows `llama.cpp` to efficiently load models, leveraging memory mapping and GPU offloading capabilities to handle large models effectively.
