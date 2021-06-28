import argparse
import logging

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


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


def load_transformer_model(model_name: str = "xlnet-base-cased", tokenizer: str = "xlnet-base-cased"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')
    return model, tokenizer


def inference(model, premise: str, hypothesis: str, tokenizer):
    log = logging.getLogger("inference")
    #log.info("Inference")
    
    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                     max_length=256, return_token_type_ids=False, truncation=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(device)
    # token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(device)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        labels=None)
        #print(outputs)
        predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one
        
        log.info(f"---------------------------------------------")
        log.info(f"Premise: {premise} - Hypothesis: {hypothesis}")
        log.info(
            f"Entailment: {predicted_probability[0]:.4f} | "
            f"Neutral: {predicted_probability[1]:.4f} | "
            f"Contradiction: {predicted_probability[2]:.4f}")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    setup_logger()
    # pretrained_model = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"
    # pretrained_model = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    pretrained_model = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    
    pairs = [("The cow is big.", "The cow chases the dog"),
             ("If the cow is big, then the cat chases the dog", "The cat chases the dog"),
             ("If the cow not is big, then the cat chases the dog", "The cat chases the dog"),
             ("The cow is big. If the cow not is big, then the cat chases the dog", "The cat chases the dog"),
             ("The cow is big", "If the cow is big, then the cat chases the dog")
             ]
    
    model, tokenizer = load_transformer_model(model_name=pretrained_model, tokenizer=pretrained_model)
    for p, h in pairs:
        inference(model=model, premise=p, hypothesis=h, tokenizer=tokenizer)
