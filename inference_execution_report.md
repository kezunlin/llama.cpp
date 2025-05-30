# Inference Execution Phase in llama.cpp

This report details the inference execution phase in the `llama.cpp` project, focusing on how tokens are generated, how the model is evaluated, and how the KV cache and sampling are managed. The analysis primarily draws from `tools/main/main.cpp`, with conceptual references to `src/llama.cpp` for `llama_decode` and `common/sampling.cpp` for sampling logic.

## 1. Main Generation Loop (`tools/main/main.cpp`)

The core of token generation in `tools/main/main.cpp` resides within a `while` loop. This loop continues as long as specific conditions are met, allowing for continuous token generation or interactive sessions.

*   **Loop Structure:**
    ```c++
    // In main()
    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // ... main generation logic ...
    }
    ```

*   **Conditions for Continuing Generation:**
    *   **`n_remain != 0`**: This variable (`params.n_predict`) tracks the number of tokens remaining to be generated. If it reaches zero (and not in interactive mode waiting for a new prediction count), the loop may terminate for non-interactive generation.
    *   **`!is_antiprompt`**: If an antiprompt sequence is detected in the generated output, `is_antiprompt` becomes true, which can stop the current generation turn (especially relevant in interactive or scripted scenarios).
    *   **`params.interactive`**: If true, the loop continues even if `n_remain` is 0, as it will typically wait for user input to start a new generation sequence or modify `n_remain`.

*   **Loop Workflow Overview:**
    1.  **Prediction/Evaluation (`if (!embd.empty())`)**: If there are tokens in the `embd` vector (either from the initial prompt, user input, or the previously generated token), this section is entered.
        *   KV cache management (context shifting or self-extend) is performed if the context is full.
        *   `llama_decode()` is called to process the tokens in `embd`.
        *   `n_past` is updated.
    2.  **Token Consumption/Sampling**:
        *   If all input tokens from `embd_inp` have been consumed (`(int) embd_inp.size() <= n_consumed`) and not in interactive mode (`!is_interacting`):
            *   A new token is sampled using `common_sampler_sample()`.
            *   The sampled token is accepted using `common_sampler_accept()`.
            *   The new token is added to `embd` for the next iteration's processing.
            *   `n_remain` is decremented.
        *   Else (if there are still tokens in `embd_inp` to process):
            *   Tokens from `embd_inp` are moved to `embd`.
            *   These prompt tokens are accepted by the sampler via `common_sampler_accept()` (typically with `accept_grammar=false`).
    3.  **Display and Output**: The generated token(s) are displayed.
    4.  **Antiprompt and EOG Handling**: Checks for antiprompts or End-of-Generation (EOG) tokens.
    5.  **Interactive Input**: If `is_interacting` is true, the system waits for user input, which is then tokenized and added to `embd_inp`.

## 2. Core Inference with `llama_decode`

The central function for running the model's forward pass is `llama_decode()`.

*   **Call in `tools/main/main.cpp`:**
    ```c++
    // In main(), inside the main generation loop and after KV cache management
    for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
        int n_eval = (int) embd.size() - i;
        if (n_eval > params.n_batch) {
            n_eval = params.n_batch;
        }

        if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return 1;
        }
        n_past += n_eval;
    }
    ```
    Here, `embd` contains the current batch of tokens to be processed (either from the prompt or previously generated ones). `llama_batch_get_one()` prepares a `llama_batch` structure for these tokens.

*   **High-Level Purpose of `llama_decode`:**
    *   `llama_decode()` (defined in `src/llama.h`, implemented in `src/llama.cpp` and `src/llama-impl.cpp`) takes the `llama_context *ctx` and a `llama_batch` of input tokens.
    *   Its primary role is to perform a forward pass of the neural network (transformer model) using these input tokens.
    *   This process updates the model's internal state (including the KV cache for the processed tokens) and produces logits (unnormalized probabilities for the next token) for the last token in the input sequence (or for all tokens if configured). These logits are stored within the `llama_context`.

*   **Computation Graph (`ggml_cgraph`) and Backend Dispatch:**
    *   Internally, `llama_decode()` constructs a computation graph (`ggml_cgraph`) using the GGML library. This graph represents all the mathematical operations (matrix multiplications, attention mechanisms, normalizations, etc.) required for the transformer's forward pass for the current batch of tokens.
    *   Once the graph is built, GGML's execution functions (e.g., `ggml_graph_compute`, `ggml_graph_compute_helper`, or backend-specific versions like `ggml_backend_graph_compute`) are called to execute the operations in the graph.
    *   **Crucially for GPU offloading:** GGML handles the dispatch of these operations. When a tensor operation is encountered in the graph:
        *   GGML checks the backend associated with the input tensor(s).
        *   If a tensor's data (`ggml_tensor->buffer`) was allocated on a GPU device during the model loading phase, the operation involving that tensor is executed on that GPU by the corresponding backend (e.g., CUDA, Metal).
        *   This ensures that computations for offloaded layers occur on the designated GPU(s), while CPU-resident tensor operations run on the CPU.

## 3. KV Cache Management (`tools/main/main.cpp`)

The Key-Value (KV) cache is essential for the efficiency of autoregressive generation in transformer models. It stores the keys and values from the self-attention mechanisms for previously processed tokens, avoiding redundant computations. `tools/main/main.cpp` implements strategies to manage this cache when the context window (`n_ctx`) fills up. `n_past` tracks the number of tokens currently in the KV cache.

*   **Context Full Check:**
    ```c++
    if (n_past + (int) embd.size() >= n_ctx) {
        // KV cache management logic is triggered
    }
    ```

*   **Context Shifting (`params.ctx_shift == true` and `ga_n == 1` i.e., no Grouped Attention):**
    *   This is the standard mechanism for making space when the context is full.
    *   A portion of the oldest tokens in the KV cache is discarded, and the remaining part is shifted.
    *   `const int n_left = n_past - params.n_keep;` (tokens to potentially discard beyond the initial `params.n_keep` tokens)
    *   `const int n_discard = n_left / 2;` (half of the non-kept tokens are discarded)
    *   `llama_kv_self_seq_rm(ctx, 0, params.n_keep, params.n_keep + n_discard);`
        *   Removes tokens from the KV cache for sequence ID 0, starting from index `params.n_keep` up to `params.n_keep + n_discard - 1`.
    *   `llama_kv_self_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);`
        *   Shifts the KV cache entries for sequence ID 0. Tokens from index `params.n_keep + n_discard` to `n_past - 1` are shifted to positions starting from `params.n_keep`. The `delta` is `-n_discard`.
    *   `n_past -= n_discard;` The effective number of tokens in the cache is reduced.

*   **Self-Extend with Grouped-Attention (`ga_n > 1`):**
    *   When Grouped-Attention Self-Extend is used, the KV cache management is different, aiming to extend the context.
    *   The loop `while (n_past >= ga_i + ga_w)` triggers modifications to the KV cache structure.
    *   `llama_kv_self_seq_add(ctx, 0, ga_i, n_past, ib*bd);`
    *   `llama_kv_self_seq_div(ctx, 0, ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n);` (divides certain parts of the KV cache, effectively compressing them)
    *   `llama_kv_self_seq_add(ctx, 0, ga_i + ib*bd + ga_w, n_past + ib*bd, dd);`
    *   `n_past -= bd;` and `ga_i += ga_w/ga_n;` update state variables for this mechanism.

## 4. Token Sampling (`tools/main/main.cpp` and `common/sampling.cpp`)

After `llama_decode()` produces logits, a sampling process selects the next token.

*   **Initialization (`tools/main/main.cpp`):**
    *   A `common_sampler` object is initialized using settings from `params.sampling` (which holds `sparams` like temperature, top-k, top-p, repetition penalties, etc.).
        ```c++
        // In main()
        auto & sparams = params.sampling;
        smpl = common_sampler_init(model, sparams);
        ```
    *   `common_sampler_init()` (in `common/sampling.cpp`) creates a chain of individual samplers (e.g., for logit bias, top-k, top-p, temperature, mirostat) based on the `sparams`. It also initializes a grammar sampler (`grmr`) if `params.grammar` is provided.

*   **Sampling Call (`tools/main/main.cpp`):**
    *   This occurs when the application is generating new tokens (i.e., not processing an initial prompt or user input).
        ```c++
        // In main(), when generating a new token
        const llama_token id = common_sampler_sample(smpl, ctx, -1);
        ```
    *   `common_sampler_sample()` (in `common/sampling.cpp`):
        1.  `gsmpl->set_logits(ctx, idx);`: It first retrieves the logits produced by `llama_decode()` from the `llama_context` (for the specified batch item `idx`, usually -1 for the last token of the sequence) and populates an internal candidate list (`gsmpl->cur_p`).
        2.  `llama_sampler_apply(chain, &cur_p);`: It applies the configured chain of samplers (temperature, top-k, top-p, penalties, etc.) to the logits in `cur_p`. Each sampler in the chain modifies the logits or selects candidates.
        3.  **Grammar Handling:** If a grammar is enabled (`gsmpl->grmr`), it first samples a token using the main chain. Then, it checks if this token conforms to the grammar. If not, it "resamples" by first applying grammar constraints to the full logit list and then applying the main sampling chain again. This ensures the final token is valid according to the grammar. The `grammar_first` parameter can alter this order.
        4.  The function returns the `llama_token id` of the selected token.

*   **Accepting Token (`tools/main/main.cpp`):**
    *   Once a token is sampled (or if processing tokens from an initial prompt), it's "accepted" to update the sampler's internal state.
        ```c++
        // When a new token 'id' is sampled:
        common_sampler_accept(smpl, id, /* accept_grammar= */ true);

        // When processing initial prompt tokens from 'embd_inp':
        common_sampler_accept(smpl, embd_inp[n_consumed], /* accept_grammar= */ false);
        ```
    *   `common_sampler_accept()` (in `common/sampling.cpp`):
        1.  `llama_sampler_accept(gsmpl->grmr, token);`: If `accept_grammar` is true, the grammar sampler is updated with the accepted token.
        2.  `llama_sampler_accept(gsmpl->chain, token);`: Each sampler in the main chain is notified of the accepted token. This is crucial for samplers that depend on history, like repetition penalties (`llama_sampler_init_penalties`), which update their internal state based on `gsmpl->prev`.
        3.  `gsmpl->prev.push_back(token);`: The accepted token is added to a ring buffer (`gsmpl->prev`) which stores a history of the last `params.n_prev` tokens. This history is used by penalty samplers.

This sequence of decoding, KV cache management, and sampling forms the core of the inference loop, enabling `llama.cpp` to generate text token by token while efficiently managing context and applying various sampling strategies.
