import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

IM_START = '<|im_start|>'
IM_END = '<|im_end|>'
STOP = '<|stop|>'
EOT = '<|endofturn|>'

def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

def resize_and_rescale_embedding(embedding, target_dim, n, std):
    """
    Resize and rescale a numpy embedding matrix.

    Args:
        embedding (np.ndarray): Original embedding matrix of shape (vocab_size, original_dim).
        target_dim (int): The target embedding dimension.
        n (float): Desired mean for the resized embedding.
        std (float): Desired standard deviation for the resized embedding.

    Returns:
        np.ndarray: Resized and rescaled embedding matrix of shape (vocab_size, target_dim).
    """
    _, original_dim = embedding.shape

    # Create a linear projection matrix to map to the target dimension
    projection_matrix = np.random.normal(0, 1, (original_dim, target_dim))

    # Resize the embedding matrix
    resized_embedding = embedding @ projection_matrix

    # Standardize the resized embedding
    current_mean = np.mean(resized_embedding)
    current_std = np.std(resized_embedding)
    standardized_embedding = (resized_embedding - current_mean) / current_std

    # Rescale to the desired mean and standard deviation
    final_embedding = standardized_embedding * std + n

    return final_embedding

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype='auto')

    prev_tokenizer_size = len(tokenizer)
    prev_embedding_size = model.get_input_embeddings().weight.data.size(0)
    embedding_dim = model.get_input_embeddings().weight.data.size(1)
    new_tokens = args.special_tokens + [args.unit_format.format(i) for i in range(args.num_units)]
    if args.add_chatml:
        new_tokens += [IM_START, IM_END, STOP, EOT]

    num_new_tokens = len(new_tokens)

    print("Previous Tokenizer Size:", prev_tokenizer_size)
    print("Previous Embedding Size:", prev_embedding_size)
    print("Number of New Tokens:", num_new_tokens)
    print("New Tokens:", " ".join(new_tokens[:3]), "...", " ".join(new_tokens[-3:]))

    # tokenizer.add_tokens(new_tokens)
    tokenizer.add_special_tokens({'additional_special_tokens': new_tokens}, 
                                 replace_additional_special_tokens=False)

    if args.from_pretrained:
        pretrained_emb = np.load(args.from_pretrained)
        if args.std == "model_config":
            in_mean, in_std = 0.0, model.config.initializer_range
            out_mean, out_std = 0.0, model.config.initializer_range
        elif args.std == "inherit":
            in_mean, in_std = torch.mean(model.get_input_embeddings().weight.data).item(), torch.std(model.get_input_embeddings().weight.data).item()
            out_mean, out_std = torch.mean(model.get_output_embeddings().weight.data).item(), torch.std(model.get_output_embeddings().weight.data).item()
        elif is_float(args.std):
            in_mean, in_std = 0.0, args.std
            out_mean, out_std = 0.0, args.std
        rescaled_in_embedding = torch.from_numpy(resize_and_rescale_embedding(pretrained_emb, embedding_dim, in_mean, in_std))
        rescaled_out_embedding = torch.from_numpy(resize_and_rescale_embedding(pretrained_emb, embedding_dim, out_mean, out_std))

        print(f"input embeddings mean, std : {in_mean}, {in_std}")
        print(f"output embeddings mean, std: {out_mean}, {out_std}")

        model.resize_token_embeddings(model.vocab_size + num_new_tokens)
        model.get_input_embeddings().weight.data[len(tokenizer) - args.num_units: len(tokenizer)] = rescaled_in_embedding
        model.get_output_embeddings().weight.data[len(tokenizer) - args.num_units: len(tokenizer)] = rescaled_out_embedding

    elif args.std == "inherit":
        model.resize_token_embeddings(model.vocab_size + num_new_tokens)
        if prev_embedding_size > prev_tokenizer_size:
            model.get_input_embeddings().weight.data = torch.cat(
                (model.get_input_embeddings().weight.data[:prev_tokenizer_size],
                 model.get_input_embeddings().weight.data[-num_new_tokens:],
                 model.get_input_embeddings().weight.data[prev_tokenizer_size:-num_new_tokens]), 
                0).contiguous()

            model.get_output_embeddings().weight.data = torch.cat(
                (model.get_output_embeddings().weight.data[:prev_tokenizer_size],
                 model.get_output_embeddings().weight.data[-num_new_tokens:],
                 model.get_output_embeddings().weight.data[prev_tokenizer_size:-num_new_tokens]), 
                0).contiguous()

    elif args.std == "model_config" or is_float(args.std):
        std = model.config.initializer_range if args.std == "model_config" else float(args.std)
        model.resize_token_embeddings(model.vocab_size + num_new_tokens)
        input_embeddings = model.get_input_embeddings().weight.data[len(tokenizer) - num_new_tokens:]
        input_embeddings.normal_(mean=0.0, std=std)
        output_embeddings = model.get_output_embeddings().weight.data[len(tokenizer) - num_new_tokens:]
        output_embeddings.normal_(mean=0.0, std=std)

    for token in new_tokens[:3]:
        print(f"added token id: {token} -> {tokenizer.convert_tokens_to_ids(token)}")
    print("...")
    for token in new_tokens[-3:]:
        print(f"added token id: {token} -> {tokenizer.convert_tokens_to_ids(token)}")    

    tokenizer.save_pretrained(args.save_path)
    model.save_pretrained(args.save_path)

    print("Save Completed:", args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/mu-jessica-000')
    parser.add_argument('--num_units', type=int, default=10000)
    parser.add_argument("--unit_format", type=str, default="<|audio{:04d}|>")
    parser.add_argument("--special_tokens", nargs='+', default="<|audio_correspond|> <|audio_continue|> <|audio_start|> <|audio_end|>")
    parser.add_argument('--from_pretrained', type=str, required=False)
    parser.add_argument('--std', type=str, default="inherit", 
        help=('"inherit" -> embeddings will be initialized from a multivariate normal distribution that has old embeddings\n'
              '"model_config" -> embeddings will be initialized from the model.config.initializer_range\n'
              'a float value (e.g. 0.02) -> custom float value will be used to initialize the embeddings'))
    parser.add_argument('--overwrite', type=bool, default=False, help="Overwrite the given model_path. Ignored if `save_path` is given.")
    parser.add_argument('--save_path', type=str, default="", help="Path to save vocab-expanded-model")
    parser.add_argument('--add_chatml', type=bool, default=False, help="Add chatml special tokens.")

    args = parser.parse_args()

    if args.save_path == "" and not args.overwrite:
        print("Should pass `--save_path={your_save_path}` or `--overwrite` option to save model.")
    elif args.save_path == "" and args.overwrite:
        args.save_path = args.model_path

    main(args)