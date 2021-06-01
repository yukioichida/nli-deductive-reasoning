import argparse
import logging

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def setup_logger():
    log_format = '%(asctime)s - %(name)s:%(levelname)s - %(message)s'
    # create logger with 'spam_application'
    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    # fh = logging.FileHandler('inference.log')
    # fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    # fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # logger.addHandler(fh)
    logger.addHandler(ch)


def load_transformer_model(model_name: str = "xlnet-base-cased", base_model_name: str = "xlnet-base-cased"):
    tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
                                              do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')
    return model, tokenizer


def inference(pretrained_model: str, premise: str, hypothesis: str):
    log = logging.getLogger("inference")
    model, tokenizer = load_transformer_model(model_name=pretrained_model)
    
    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                     max_length=256, return_token_type_ids=True, truncation=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(device)
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(device)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(device)
    
    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)
    
    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one
    
    log.info(f"Premise: {premise} - Hypothesis: {hypothesis}")
    log.info(f"Entailment: {predicted_probability[0]:.4f}")
    log.info(f"Neutral: {predicted_probability[1]:.4f}")
    log.info(f"Contradiction: {predicted_probability[2]:.4f}")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference in an NLI Model')
    parser.add_argument('--pretrained_model', type=str, required=True, help='Pretrained dir model')
    parser.add_argument('--h', type=str, help='Hypothesis')
    parser.add_argument('--p', type=str, help='Premise')
    parser.add_argument('--seed', type=int, default=42, help='SEED')
    
    args = parser.parse_args()
    setup_logger()
    set_seed(args.seed)
    logging.getLogger().info(args)
    inference(pretrained_model=args.pretrained_model, premise=args.p, hypothesis=args.h)
