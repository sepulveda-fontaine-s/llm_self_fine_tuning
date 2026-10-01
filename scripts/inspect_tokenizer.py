from transformers import AutoConfig, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def main() -> None:
    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("=== MODEL ===")
    print("model:", MODEL_NAME)
    print("architecture:", config.architectures)
    print("model_type:", config.model_type)
    print("max_position_embeddings:", config.max_position_embeddings)

    print("\n=== TOKENIZER ===")
    print("class:", tokenizer.__class__.__name__)
    print("vocab_size:", len(tokenizer))
    print("model_max_length:", tokenizer.model_max_length)
    print("padding_side:", tokenizer.padding_side)

    print("\n=== SPECIAL TOKENS ===")
    print("bos_token:", repr(tokenizer.bos_token), tokenizer.bos_token_id)
    print("eos_token:", repr(tokenizer.eos_token), tokenizer.eos_token_id)
    print("pad_token:", repr(tokenizer.pad_token), tokenizer.pad_token_id)
    print("unk_token:", repr(tokenizer.unk_token), tokenizer.unk_token_id)

    text = (
        "Self-supervised language-model training predicts the next token "
        "from the preceding context."
    )

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
    )

    print("\n=== TOKENIZATION SAMPLE ===")
    print("text:", text)
    print("token_count:", len(encoded["input_ids"]))
    print("input_ids:", encoded["input_ids"])
    print("tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
    print("decoded:", tokenizer.decode(encoded["input_ids"]))


if __name__ == "__main__":
    main()