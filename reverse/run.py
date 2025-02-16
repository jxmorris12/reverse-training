import transformers

from gcg import run

def limit_layers(model: transformers.PreTrainedModel, n_layers: int) -> transformers.PreTrainedModel:
    if hasattr(model, 'transformer'):
        if hasattr(model.transformer, 'h'):
            # gpt2
            model.transformer.h = model.transformer.h[:n_layers]
        else:
            model.transformer.layer = model.transformer.layer[:n_layers]
    elif hasattr(model, 'encoder'):
        if hasattr(model.encoder, 'layers'):
            model.encoder.layers = model.encoder.layers[:n_layers]
        else:
            model.encoder.layer = model.encoder.layer[:n_layers]
    else:
        raise RuntimeError(f"unknown how to limit layers of model {type(model)}")
    return model


def main():
    gpt2 = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    initial_model = limit_layers(gpt2, 4)
    final_model = transformers.AutoModelForCausalLM.from_pretrained("/home/jxm3/research/reverse-training/train/saves/checkpoint-468")
    final_model = limit_layers(final_model, 4)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    run(
        initial_model=initial_model, 
        final_model=final_model, 
        tokenizer=tokenizer, 
    )

if __name__ == "__main__": main()