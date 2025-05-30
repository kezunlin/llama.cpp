# Data Preparation Phase in llama.cpp

This report details the data preparation phase in the `llama.cpp` project, focusing on how input is acquired, tokenized, formatted, and how special tokens are handled. The analysis is based on `tools/main/main.cpp` and `common/common.cpp`.

## 1. Input Acquisition

Input text for the language model can be provided through several mechanisms:

**Command-Line Arguments:**

The `main()` function in `tools/main/main.cpp` utilizes the `common_params_parse()` function (from `common/common.cpp`, though its internal workings are not detailed here) to parse command-line arguments. Key arguments for input text include:

*   **`-p "PROMPT"` or `--prompt "PROMPT"`:** Specifies the main prompt text directly on the command line. This is stored in `params.prompt`.
*   **`-f FILENAME` or `--file FILENAME`:** Specifies a file from which to read the main prompt text. The content of this file is loaded into `params.prompt`.
*   **`--system-prompt "SYSTEM_PROMPT"` or `-sys "SYSTEM_PROMPT"`:** Specifies a system message or initial context for the model, particularly used in chat or conversational modes. This is stored in `params.system_prompt`. The `main.cpp` code shows:
    ```c++
    // In main()
    if (params.conversation_mode && params.enable_chat_template) {
        if (!params.system_prompt.empty()) {
            // format the system prompt (will use template default if empty)
            chat_add_and_format("system", params.system_prompt);
        }
        // ...
    }
    ```

**Interactive Input:**

`tools/main/main.cpp` supports an interactive mode where the user can provide input incrementally.

*   Interactive mode is enabled if `params.interactive` is true (e.g., set by `params.interactive_first` or if no initial prompt is given in conversational mode).
*   The core interactive loop is within the main `while` loop:
    ```c++
    // In main() while loop, when is_interacting is true
    if ((n_past > 0 || waiting_for_first_input) && is_interacting) {
        // ...
        std::string buffer;
        // ...
        do {
            another_line = console::readline(line, params.multiline_input);
            buffer += line;
        } while (another_line);
        // ...
        // Process the buffer
        if (params.conversation_mode && params.enable_chat_template) {
            user_inp = chat_add_and_format("user", std::move(buffer));
        } else {
            user_inp = std::move(buffer);
        }
        // ...
        const auto line_inp = common_tokenize(ctx, user_inp, false, format_chat);
        embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
        // ...
    }
    ```
*   The `console::readline()` function is responsible for capturing user input line by line.
*   Input can be multi-line if `params.multiline_input` is enabled.

## 2. Tokenization

Once the raw input text (from prompt, file, system prompt, or interactive input) is acquired, it needs to be converted into a sequence of tokens that the model can understand.

**`common_tokenize()`:**

*   This function, found in `common/common.cpp`, is the primary interface used by `tools/main/main.cpp` for tokenization.
    ```c++
    // From common/common.cpp
    std::vector<llama_token> common_tokenize(
      const struct llama_context * ctx,
               const std::string & text,
                            bool   add_special,
                            bool   parse_special) {
        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        return common_tokenize(vocab, text, add_special, parse_special);
    }

    std::vector<llama_token> common_tokenize(
        const struct llama_vocab * vocab,
               const std::string & text,
                            bool   add_special,
                            bool   parse_special) {
        // upper limit for the number of tokens
        int n_tokens = text.length() + 2 * add_special;
        std::vector<llama_token> result(n_tokens);
        n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        // ... (resize logic) ...
        return result;
    }
    ```
*   **`add_special` parameter:** When `true`, this parameter instructs `llama_tokenize` to add special control tokens (like BOS/EOS, if the vocabulary is configured for them) automatically to the beginning or end of the tokenized output. In `tools/main/main.cpp`, it's often set to `true` for the initial prompt tokenization:
    ```c++
    // In main.cpp, initial prompt tokenization
    embd_inp = common_tokenize(ctx, prompt, true, true);
    ```
    For interactive inputs or specific parts like antiprompts, it's often `false` because special tokens are handled more explicitly or aren't desired.
*   **`parse_special` parameter:** When `true`, this parameter allows `llama_tokenize` to interpret and convert special token strings (e.g., `<|system|>`) found within the input text into their corresponding token IDs. This is particularly relevant when using chat templates that might include such special tokens directly in the formatted string. In `tools/main/main.cpp`, this is generally `true` when tokenizing prompts that might have been processed by chat templating. For example, when `format_chat` (which is `params.conversation_mode && params.enable_chat_template`) is true during interactive input:
    ```c++
    // In main.cpp, interactive input tokenization
    const auto line_inp = common_tokenize(ctx, user_inp, false, format_chat);
    ```

**`llama_tokenize()`:**

*   `common_tokenize()` itself calls `llama_tokenize()`.
*   `llama_tokenize()` is a core library function, declared in `src/llama.h` (implementation likely in `src/llama-vocab.cpp` or a similar file focused on vocabulary and tokenization logic). This function performs the actual conversion of text to token IDs based on the model's vocabulary.

## 3. Prompt Formatting and Chat Templating

`tools/main/main.cpp` includes sophisticated logic for constructing the final prompt, especially when dealing with conversations and chat templates.

**Initial Prompt Construction:**

*   The initial prompt string (`std::string prompt`) is assembled based on several factors:
    *   If not in conversation mode or if chat templates are disabled, `params.prompt` (from `-p` or `-f`) is used directly.
    *   If `params.conversation_mode` is enabled AND `params.enable_chat_template` is true:
        *   `common_chat_templates_init()` is called to initialize chat template data from the model or a user-specified template string (`params.chat_template`).
            ```c++
            // In main.cpp
            auto chat_templates = common_chat_templates_init(model, params.chat_template);
            ```
        *   If `params.system_prompt` is provided, it's formatted as the first message with the "system" role using `chat_add_and_format()` (which internally uses `common_chat_format_single()`):
            ```c++
            // In main.cpp
            auto chat_add_and_format = [&chat_msgs, &chat_templates](const std::string & role, const std::string & content) {
                common_chat_msg new_msg;
                new_msg.role = role;
                new_msg.content = content;
                auto formatted = common_chat_format_single(chat_templates.get(), chat_msgs, new_msg, role == "user", g_params->use_jinja);
                chat_msgs.push_back(new_msg);
                return formatted;
            };
            // ...
            if (!params.system_prompt.empty()) {
                chat_add_and_format("system", params.system_prompt);
            }
            ```
        *   If `params.prompt` (initial user prompt) is provided, it's formatted and appended with the "user" role using `chat_add_and_format()`.
        *   Finally, `common_chat_templates_apply()` is called to assemble the complete prompt string from all formatted messages:
            ```c++
            // In main.cpp
            if (!params.system_prompt.empty() || !params.prompt.empty()) {
                common_chat_templates_inputs inputs;
                inputs.messages = chat_msgs;
                inputs.add_generation_prompt = !params.prompt.empty(); // Controls if assistant generation prefix is added

                prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            }
            ```

**Conversation Mode (`params.conversation_mode`):**

*   When `params.conversation_mode` is active, `tools/main/main.cpp` maintains a history of the conversation in `chat_msgs` (a `std::vector<common_chat_msg>`).
*   During interactive input:
    *   The user's new input is captured.
    *   `chat_add_and_format()` is called with the role "user" and the new input. This function uses `common_chat_format_single()` to apply the chat template to this new user message, considering the existing `chat_msgs` history. The result is the string that gets tokenized.
        ```c++
        // In main.cpp, interactive loop
        bool format_chat = params.conversation_mode && params.enable_chat_template;
        std::string user_inp = format_chat
            ? chat_add_and_format("user", std::move(buffer)) // buffer is user's raw text
            : std::move(buffer);
        const auto line_inp = common_tokenize(ctx, user_inp, false, format_chat);
        ```
    *   After the model generates a response, the assistant's reply is collected in `assistant_ss`. If an EOG token is found or interaction is triggered, `chat_add_and_format()` is called with the role "assistant" and the collected response to add it to `chat_msgs`.
        ```c++
        // In main.cpp, after model generation, if EOG or interaction
        if (params.enable_chat_template) {
            chat_add_and_format("assistant", assistant_ss.str());
        }
        ```

**`common/common.cpp` Chat Template Functions:**

*   **`common_chat_templates_init(model, template_str)`:** Initializes the `common_chat_templates` structure. It tries to load the template from the model's metadata. If `template_str` is provided, it overrides the model's default.
*   **`common_chat_format_single(templates, messages, new_message, is_user, use_jinja)`:** Takes the current message history (`messages`), the new message to format (`new_message`), and formats this single new message according to the chat template. It considers whether the new message is from the "user" to potentially add necessary prefixes/suffixes for user turns.
*   **`common_chat_templates_apply(templates, inputs)`:** Takes a `common_chat_templates_inputs` structure (which includes all `chat_msgs` and a flag `add_generation_prompt`). It iterates through the messages, applies the template to each, and concatenates them into a single string, also adding the "assistant generation prompt" if specified. This resulting string is what gets tokenized.

**Input Prefixes/Suffixes:**

*   `params.input_prefix` and `params.input_suffix` are strings that can be added before and after user input, respectively, primarily in interactive mode when not using full chat templating.
    ```c++
    // In main.cpp, interactive loop
    // ...
    if (!params.input_prefix.empty() && !params.conversation_mode) { // Note: not applied if conversation_mode with chat_template is on
        LOG_DBG("appending input prefix: '%s'\n", params.input_prefix.c_str());
        // ... tokenization of prefix happens here ...
    }
    // ...
    const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true); // Tokenized separately
    const auto line_inp = common_tokenize(ctx, user_inp,            false, format_chat);
    const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true); // Tokenized separately

    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());
    ```
*   If `params.enable_chat_template` is false (even in conversation mode), these prefixes/suffixes might be used. However, the primary mechanism for structuring conversational input is the chat template system. The code notes a TODO: "one inconvenient of current chat template implementation is that we can't distinguish between user input and special tokens (prefix/postfix)".

## 4. Special Token Handling

**Beginning-of-Sequence (BOS):**

*   The `add_bos` flag is determined by `llama_vocab_get_add_bos(vocab) && !params.use_jinja`. The `llama_vocab_get_add_bos(vocab)` function (core library) indicates if the model's vocabulary is configured to expect a BOS token. It's also explicitly not added if `params.use_jinja` (Jinja templating for chat) is true, as Jinja templates are expected to handle BOS/EOS themselves.
*   The initial prompt tokenization (`embd_inp = common_tokenize(ctx, prompt, true, true);`) will add BOS if `add_bos` is true and the tokenizer deems it appropriate (due to the `add_special=true` argument).
*   In interactive mode, `params.input_prefix_bos` can force the addition of a BOS token before user input:
    ```c++
    // In main.cpp, interactive loop
    if (params.input_prefix_bos) {
        LOG_DBG("adding input prefix BOS token\n");
        embd_inp.push_back(llama_vocab_bos(vocab));
    }
    ```
*   If the input `embd_inp` is empty after initial processing and `add_bos` is true, a BOS token is added:
    ```c++
    // In main.cpp
    if (!waiting_for_first_input && embd_inp.empty()) {
        if (add_bos) {
            embd_inp.push_back(llama_vocab_bos(vocab));
            // ...
        }
    }
    ```

**End-of-Sequence (EOS/EOG):**

*   The `llama_vocab_is_eog(vocab, token_id)` function (core library) checks if a given token ID is an End-of-Generation token (this can include EOS or other specific EOG tokens defined by the vocabulary).
*   During generation, if `llama_vocab_is_eog()` returns true for the last sampled token and the mode is not interactive, generation stops.
    ```c++
    // In main.cpp, main generation loop
    if (!embd.empty() && llama_vocab_is_eog(vocab, embd.back()) && !(params.interactive)) {
        LOG(" [end of text]\n");
        break;
    }
    ```
*   In interactive mode with chat templates, if an EOG token is detected, it signals the end of the assistant's turn. The assistant's message is then formatted and added to the chat history.
    ```c++
    // In main.cpp, after sampling a token
    if (!waiting_for_first_input && llama_vocab_is_eog(vocab, common_sampler_last(smpl))) {
        // ...
        if (params.enable_chat_template) {
            chat_add_and_format("assistant", assistant_ss.str());
        }
        is_interacting = true;
        // ...
    }
    ```
*   There's also `need_insert_eot` (End of Turn) logic. If a user interrupts (Ctrl+C) during assistant generation in chat mode, `need_insert_eot` becomes true. Then, before the next user input is tokenized, an EOT token (or EOS if EOT is not available) is inserted to properly terminate the assistant's partial response in the context.
    ```c++
    // In main.cpp, interactive input processing
    if (need_insert_eot && format_chat) { // format_chat implies enable_chat_template
        llama_token eot = llama_vocab_eot(vocab);
        embd_inp.push_back(eot == LLAMA_TOKEN_NULL ? llama_vocab_eos(vocab) : eot);
        need_insert_eot = false;
    }
    ```
    The `llama_vocab_eot(vocab)` function likely retrieves the specific End-of-Turn token ID if defined by the model's vocabulary.

This covers the main aspects of data preparation in `llama.cpp`'s main example. The system is flexible, allowing for simple prompts, file-based inputs, and complex, templated chat interactions.
