import gradio as gr

MODEL_NAME = "Forge-20M"
MODEL_TAGLINE = "A 20.75M parameter language model trained from scratch"

n_params = sum(p.numel() for p in model.parameters()) / 1e6

# -----------------------------------------------------------------------------
# STREAMING GENERATION — yields one word/token at a time
# -----------------------------------------------------------------------------
def stream_generate_with_penalty(prompt, max_new_tokens=150, temperature=0.7, top_k=25, repetition_penalty=1.3):
    start_ids = enc.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    idx = x
    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]

            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next.item() == 50256:
                break

            idx = torch.cat((idx, idx_next), dim=1)
            generated_ids.append(idx_next.item())

            # yield the decoded text so far after each new token
            yield enc.decode(generated_ids)


def respond(message, history, max_new_tokens, temperature, top_k, repetition_penalty):
    if not message or not message.strip():
        yield "", history
        return

    history = history + [{"role": "user", "content": message},
                          {"role": "assistant", "content": ""}]

    for partial_text in stream_generate_with_penalty(
        message,
        max_new_tokens=int(max_new_tokens),
        temperature=temperature,
        top_k=int(top_k),
        repetition_penalty=repetition_penalty,
    ):
        history[-1]["content"] = partial_text
        yield "", history


def clear_chat():
    return None, ""

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#FFFDF9",
    body_background_fill_dark="#FFFDF9",
    block_background_fill="#FFFFFF",
    block_border_color="#F0E4D8",
    block_border_width="1px",
    block_shadow="none",
    button_primary_background_fill="#E8A87C",
    button_primary_background_fill_hover="#D9926A",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_border_color="#E5D5C3",
    button_secondary_text_color="#5A4A3A",
    input_background_fill="#FFFFFF",
    input_border_color="#E5D5C3",
    body_text_color="#3A2E24",
    body_text_color_subdued="#8A7A68",
)

CUSTOM_CSS = """
#header { text-align: center; padding: 20px 0 8px 0; }
#header h1 { margin-bottom: 4px; font-weight: 600; color: #3A2E24; }
#header p { color: #9C8B78; margin-top: 0; font-size: 0.95em; }
#chat-column { min-height: 600px; }
.footnote { text-align: center; font-size: 0.8em; color: #B0A090; margin-top: 12px; }
"""

with gr.Blocks(title=MODEL_NAME, theme=theme, css=CUSTOM_CSS) as demo:

    gr.Markdown(
        f"# {MODEL_NAME}\n{MODEL_TAGLINE}  \n"
        f"{n_params:.1f}M parameters running on {device.upper()}",
        elem_id="header",
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### Generation settings")
            max_new_tokens = gr.Slider(16, 300, value=150, step=1, label="Max new tokens")
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
            top_k = gr.Slider(1, 200, value=25, step=1, label="Top-k")
            repetition_penalty = gr.Slider(1.0, 2.0, value=1.3, step=0.05, label="Repetition penalty")
            clear_btn = gr.Button("Clear conversation", variant="secondary")

        with gr.Column(scale=3, elem_id="chat-column"):
            chatbot = gr.Chatbot(
                height=560,
                type="messages",
                show_copy_button=True,
                avatar_images=(None, None),
                label=None,
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type a prompt and press Enter",
                    show_label=False,
                    scale=8,
                    lines=1,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

    gr.Markdown(
        f"<div class='footnote'>{MODEL_NAME} is a small next word prediction model. "
        f"Outputs may be incoherent or factually inaccurate.</div>"
    )

    inputs = [msg, chatbot, max_new_tokens, temperature, top_k, repetition_penalty]

    msg.submit(respond, inputs, [msg, chatbot])
    send_btn.click(respond, inputs, [msg, chatbot])
    clear_btn.click(clear_chat, None, [chatbot, msg])

demo.queue()
demo.launch(share=True)