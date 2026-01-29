import logging
import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
from huggingface_hub import hf_hub_download
from clemcore import backends
from clemcore.backends.utils import ensure_alternating_roles, ensure_messages_format, augment_response_object
from typing import List, Dict, Tuple, Any
import re

logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.cli")

def load_system_prompt(repo_id: str, filename: str = 'SYSTEM_PROMPT.txt') -> dict[str, Any]:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()

    index_begin_think = system_prompt.find("[THINK]")
    index_end_think = system_prompt.find("[/THINK]")

    return {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt[:index_begin_think]},
            {
                "type": "thinking",
                "thinking": system_prompt[
                    index_begin_think + len("[THINK]") : index_end_think
                ],
                "closed": True,
            },
            {
                "type": "text",
                "text": system_prompt[index_end_think + len("[/THINK]") :],
            },
        ],
    }

def load_config_and_tokenizer(model_spec: backends.ModelSpec):
    """Load model config and tokenizer from Huggingface.

    Args:
        model_spec: A ModelSpec instance specifying the model.

    Returns:
        tokenizer: The loaded tokenizer.
        config: The loaded model configuration.
        context_size: The model's context size.
    """
    hf_model_str = model_spec['huggingface_id']
    tokenizer = MistralCommonBackend.from_pretrained(hf_model_str)
    auto_config = Mistral3ForConditionalGeneration.from_pretrained(hf_model_str).config
    context_size = auto_config.text_config.max_position_embeddings
    padding_side = model_spec.model_config.get("padding_side", None)
    if padding_side is None:
        stdout_logger.warning("No 'padding_side' configured in 'model_config' for %s", model_spec.model_name)
        tokenizer.padding_side = "left" if auto_config.is_decoder and not auto_config.is_encoder_decoder else "right"
        stdout_logger.warning("Derive padding_size=%s from model architecture (decoder=%s, encoder-decoder=%s)",
                              tokenizer.padding_side, auto_config.is_decoder, auto_config.is_encoder_decoder)
    else:
        padding_side = padding_side.lower()
        if padding_side not in ("left", "right"):
            raise ValueError(f"Invalid 'padding_side={padding_side}' configured in 'model_config' "
                             f"for {model_spec.model_name}. Must be 'left' or 'right'.")
        tokenizer.padding_side = padding_side
    return tokenizer, auto_config, context_size

def load_model(model_spec: backends.ModelSpec):
    """Load model from Huggingface.

    Args:
        model_spec: A ModelSpec instance specifying the model.
    Returns:
        model: The loaded model.
    """
    hf_model_str = model_spec['huggingface_id']
    model = Mistral3ForConditionalGeneration.from_pretrained(
        hf_model_str, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model

class MistralLocal(backends.Backend):
    """Model/backend handler class for locally-run Huggingface models."""

    def __init__(self):
        super().__init__()

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """Get a MistralLocalModel instance with the passed model and settings.
        Will load all required data for using the model upon initialization.
        Args:
            model_spec: The ModelSpec for the model.
        Returns:
            The Model class instance of the model.
        """
        torch.set_num_threads(1)
        return MistralLocalModel(model_spec)

class MistralLocalModel(backends.BatchGenerativeModel):
    """Mistral Local Model Backend using Huggingface Transformers"""

    def __init__(self, model_spec: backends.ModelSpec):
        """
        Args:
            model_spec: A ModelSpec instance specifying the model.
        """
        super().__init__(model_spec)
        # fail-fast
        self.tokenizer, self.config, self.context_size = load_config_and_tokenizer(model_spec)
        self.model = load_model(model_spec)

        # check if model's generation_config has pad_token_id set:
        if not self.model.generation_config.pad_token_id:
            # set pad_token_id to tokenizer's eos_token_id to prevent excessive warnings:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if 'thinking' in self.model_spec.model_config and self.model_spec.model_config['thinking']:
            # thinking mode is only enabled by prepending system prompt
            self.system_prompt = load_system_prompt(
                repo_id=model_spec.huggingface_id
            )
    
    @augment_response_object
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        """
        Public method for single response generation.

        Wraps the batch response method internally to reuse batch logic.

        Args:
            messages (List[Dict]): List of message dicts.

        Returns:
            Tuple[Any, Any, str]: Single response tuple (prompt, response_object, response_text).
        """
        batch_messages = [messages]  # Wrap single message list into batch
        # Call batch method without decorators to avoid double invocation of decorators
        results = self._generate_batch_response(batch_messages)

        return results[0]  # Unpack single result to maintain original API
    
    @augment_response_object
    @ensure_messages_format
    def generate_batch_response(self, batch_messages: List[List[Dict]]) -> List[Tuple[Any, Any, str]]:
        """
        Public method for batch response generation.

        Args:
            batch_messages (List[List[Dict]]): Batch of message lists.

        Returns:
            List[Tuple[Any, Any, str]]: List of response tuples.
        """
        return self._generate_batch_response(batch_messages)

    def _generate_batch_response(self, batch_messages: List[List[Dict]]) -> List[Tuple[Any, Any, str]]:
        """
        Core batch response implementation without decorators.

        Args:
            batch_messages (List[List[Dict]]): Batch of message lists,
                assumed to be properly formatted.

        Returns:
            List[Tuple[Any, Any, str]]: List of response tuples (prompt, response_object, response_text).

        Note:
            Intended for internal use only. Use public decorated methods
            for normal calls to ensure formatting and metadata.
        """

        gen_args = gen_args = {
            "do_sample": False,
            "temperature": None,  # avoid warning
            "top_p": None,  # avoid warning
            "max_new_tokens": self.max_tokens,
        }

        if 'thinking' in self.model_spec.model_config and self.model_spec.model_config['thinking']:
            # prepend system prompt if not disabled in model spec
            for messages in batch_messages:
                messages.insert(0, self.system_prompt)
            gen_args["max_new_tokens"] = self.context_size

        rendered_chats = self.tokenizer.apply_chat_template(
            batch_messages, 
            return_tensors="pt", 
            return_dict=True,
            padding=True).to(self.model.device)
        prompt_token_ids = rendered_chats["input_ids"].to(device=self.model.device)

        # Check context limit for each input in the batch
        assert_context_limits(self, prompt_token_ids)

        if self.temperature > 0.0:
            gen_args["do_sample"] = True
            gen_args["top_p"] = getattr(self.model.generation_config, "top_p", 0.95)  # look in config for default value
            gen_args["temperature"] = self.temperature
        
        model_output_ids = self.model.generate(
            **rendered_chats,
            **gen_args
        )

        # Decode all outputs and prompts
        model_outputs = self.tokenizer.batch_decode(model_output_ids)
        prompt_texts = self.tokenizer.batch_decode(prompt_token_ids)

        prompts, response_texts, responses = split_and_clean_batch_outputs(self,
                                                                           model_outputs,
                                                                           prompt_texts)
        return list(zip(prompts, responses, response_texts))
    
def split_and_clean_batch_outputs(model: MistralLocalModel,
                                  model_outputs: List[str],
                                  prompt_texts: List[str]) -> Tuple[List[str], List[str], List[Any]]:
    """
    Processes a batch of raw model output strings by removing input prompts,
    trimming any configured output prefixes, and cleaning up end-of-sequence tokens.

    Args:
        model: The HuggingfaceLocalModel instance containing model configuration and settings.
        model_outputs: List of raw generated output strings from the model (batch).
        prompt_texts: List of prompt strings corresponding to each model output in the batch.

    Returns:
        Tuple of three lists (prompts, response_texts, responses):
        - prompts: List of dicts with prompt information (inputs, max_new_tokens, temperature, etc.).
        - response_texts: List of cleaned response strings, with prompts removed and special tokens trimmed.
        - responses: List of dicts containing the raw model output strings under the key 'response'.
    """
    prompts = []
    responses = []
    response_texts = []

    for model_output, prompt_text in zip(model_outputs, prompt_texts):
        # Remove prompt from output
        response_text = model_output.replace(prompt_text, '').strip()
        # Remove batch processing padding tokens
        if response_text.startswith(model.tokenizer.pad_token) or response_text.endswith(model.tokenizer.pad_token):
            response_text = response_text.replace(model.tokenizer.pad_token, "").strip()
        # Remove EOS tokens and potential trailing tokens from response
        eos_to_cull = model.model_spec.model_config['eos_to_cull']  # This is a regEx to handle inconsistent outputs
        response_text = re.sub(eos_to_cull, "", response_text)

        # Check for CoT output and split if present
        if 'thinking' in model.model_spec.model_config and model.model_spec.model_config['thinking']:
            cot_content, response_text = split_and_clean_cot_output(response_text, model)

        prompt_info = {
            "inputs": prompt_text,
            "max_new_tokens": model.max_tokens,
            "temperature": model.temperature,
        }
        response_info = {
            "response": model_output,
        }
        # Add cot_content content to response_info
        if 'thinking' in model.model_spec.model_config and model.model_spec.model_config['thinking']:
            response_info['cot_content'] = cot_content

        prompts.append(prompt_info)
        responses.append(response_info)
        response_texts.append(response_text)
    
    return prompts, response_texts, responses

def split_and_clean_cot_output(response_text: str, model: MistralLocalModel) -> Tuple[str, str]:
    """
    Splits and cleans the chain-of-thought (CoT) content from the response text.

    Args:
        response_text: The raw response text potentially containing CoT content.
        model: The MistralLocalModel instance containing model configuration.
    Returns:
        Tuple containing:
            - cot_content: The extracted CoT content.
            - cleaned_response_text: The response text with CoT content removed.
    """
    cot_start_tag = model.model_spec.model_config.get('cot_start_tag', '\[THINK\]')
    response_text = response_text.replace(cot_start_tag, "")

    cot_end_tag = model.model_spec.model_config.get('cot_end_tag', '\[/THINK\]')
    split_cot_response = re.split(cot_end_tag, response_text)

    cot_content = split_cot_response[0]
    # Handle empty CoT outputs
    if len(split_cot_response) >= 2:
        answer = split_cot_response[-1]
    else:
        answer = ""

    # Retokenize and count CoT and final answer tokens
    tokenized_answer = model.tokenizer(answer)
    tokenized_answer = tokenized_answer.input_ids
    n_answer_tokens = len(tokenized_answer)
    # Cut answer tokens to max_tokens value if they exceed it
    if n_answer_tokens > model.max_tokens:
        logger.info(f"CoT final answer token count {n_answer_tokens} exceeds max_tokens {model.max_tokens}, "
                    f"cutting off excess tokens.")
        tokenized_answer = tokenized_answer[:model.max_tokens]
    
    # Decode retokenized and potentially cut answer
    answer = model.tokenizer.decode(tokenized_answer, skip_special_tokens=True)
    # Strip answer to assure proper clemgame parsing
    answer = answer.strip()

    return cot_content, answer

def assert_context_limits(model: MistralLocalModel, prompt_token_ids):
    for i in range(prompt_token_ids.size(0)):
        context_check = _check_context_limit(
            model.context_size,
            prompt_token_ids[i],
            max_new_tokens=model.max_tokens
        )
        if not context_check[0]:
            logger.info(f"Context token limit for {model.model_spec.model_name} exceeded on batch index {i}: "
                        f"{context_check[1]}/{context_check[3]}")
            raise backends.ContextExceededError(
                f"Context token limit for {model.model_spec.model_name} exceeded at batch index {i}",
                tokens_used=context_check[1],
                tokens_left=context_check[2],
                context_size=context_check[3]
            )


def _check_context_limit(context_size, prompt_tokens, max_new_tokens: int = 100) -> Tuple[bool, int, int, int]:
    """Internal context limit check to run in generate_response.
    Args:
        prompt_tokens: List of prompt token IDs.
        max_new_tokens: How many tokens to generate ('at most', but no stop sequence is defined).
    Returns:
        Tuple with
            Bool: True if context limit is not exceeded, False if too many tokens
            Number of tokens for the given messages and maximum new tokens
            Number of tokens of 'context space left'
            Total context token limit
    """
    prompt_size = len(prompt_tokens)
    tokens_used = prompt_size + max_new_tokens  # context includes tokens to be generated
    tokens_left = context_size - tokens_used
    fits = tokens_used <= context_size
    return fits, tokens_used, tokens_left, context_size

