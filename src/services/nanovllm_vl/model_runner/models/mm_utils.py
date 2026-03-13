import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable, Tuple

from src.services.nanovllm_vl.engine.mm_io_struct import ImageDataItem, ImageInputs
from src.services.nanovllm_vl.engine.multimodal_cache import MultiModalCache
from src.services.nanovllm_vl.utils.context import get_context, set_context_field

def flatten_nested_list(nested_list):
    if isinstance(nested_list, list):
        return [
            item for sublist in nested_list for item in flatten_nested_list(sublist)
        ]
    else:
        return [nested_list]

def _get_precomputed_embedding(
    items: List[ImageDataItem],
) -> Optional[torch.Tensor]:
    """
    If all items have precomputed_embeddings, return their concatenation.
    If some but not all have precomputed_embeddings, raise NotImplementedError.
    If none have precomputed_embeddings, return None.
    """
    precomputed_embeddings = [item.precomputed_embeddings for item in items]
    if any(feature is not None for feature in precomputed_embeddings):
        if not all(feature is not None for feature in precomputed_embeddings):
            raise NotImplementedError(
                "MM inputs where only some items are precomputed."
            )
        result = torch.concat(precomputed_embeddings)
        # some models embedding is 3-dim, reshape it to 2-dim (similar to get_embedding_chunk)
        result = result.reshape(-1, result.shape[-1])
        return result
    return None

embedding_cache: Optional[MultiModalCache] = None

def init_embedding_cache(max_size: int = 0):
    global embedding_cache
    embedding_cache = MultiModalCache(max_size)


def get_embedding_hash(embedding_items: List[ImageDataItem]) -> int:
    hash_list = [item.hash for item in embedding_items]
    return hash(tuple(hash_list))

def get_embedding_chunk(
    embedding: torch.Tensor,
    extend_prefix_len: int,
    extend_seq_len: int,
    items_offset: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, int, int]:
    """
    Extract a chunk of embeddings based on the specified prefix length, sequence length, and offset ranges.

    Args:
        embedding: The full embedding tensor to extract a chunk from
        extend_prefix_len: The starting position (prefix length) for extraction
        extend_seq_len: The number of tokens to extract
        items_offset: List of [start, end] offset ranges for multimodal items in the input sequence

    Returns:
        A tuple containing:
        - The extracted embedding chunk as a tensor
        - The start index used for extraction
        - The end index used for extraction

    Note:
        If there's no overlap between the requested range and the offset ranges,
        an empty tensor is returned with zeros for start and end indices.
    """
    start_index, end_index = 0, 0
    extend_start_index = extend_prefix_len
    extend_end_index = extend_prefix_len + extend_seq_len - 1

    for start, end in items_offset:
        if extend_start_index >= start and extend_start_index <= end:
            start_index += extend_start_index - start
        elif extend_start_index > end:
            start_index += end - start + 1

        if extend_end_index >= start and extend_end_index <= end:
            end_index += extend_end_index - start + 1
        elif extend_end_index > end:
            end_index += end - start + 1
    # some models' embedding is 3-dim, reshape it to 2-dim
    embedding = embedding.reshape(-1, embedding.shape[-1])
    embedding_chunk = embedding[start_index:end_index]
    return embedding_chunk, start_index, end_index

def _get_chunked_prefill_embedding(
    data_embedding_func: Callable[[List[ImageDataItem]], torch.Tensor],
    embedding_items: List[ImageDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Optional[torch.Tensor]:
    # Calculate embedding for each request, try to get it from cache to avoid repeated calculation
    embedding_list = []
    # FIXME(Xinyuan): temporary workaround for eagle3, which may have len(items_size) > len(prefix_length)
    max_iterations = min(len(items_size) - 1, len(prefix_length))
    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        assert items_offset is not None, items_offset
        embedding_items_hash = get_embedding_hash(embedding_items_per_req)
        # if all items has been prefixed, we do not need to calculate embedding
        if all([offset_end < prefix_length[i] for _, offset_end in items_offset]):
            continue
        embedding_per_req = embedding_cache.get(embedding_items_hash)
        if embedding_per_req is None:
            embedding_per_req = data_embedding_func(embedding_items_per_req)
            if not embedding_cache.put(embedding_items_hash, embedding_per_req):
                print(
                    "Multimodal embedding cache is full. This typically occurs when a single "
                    "embedding exceeds the cache size limit. Consider increasing the "
                    "`SGLANG_VLM_CACHE_SIZE_MB` environment variable or reducing the input "
                    "embedding size."
                )

        embedding_per_req_chunk, _, _ = get_embedding_chunk(
            embedding=embedding_per_req,
            extend_prefix_len=prefix_length[i],
            extend_seq_len=extend_length[i] if i < len(extend_length) else 0,
            items_offset=items_offset,
        )
        
        embedding_list.append(embedding_per_req_chunk)
    if len(embedding_list) == 0:
        return None
    return torch.concat(embedding_list, dim=0)

def _get_multimodal_mask(
    input_ids: torch.Tensor, placeholder_tensor: torch.Tensor
) -> torch.Tensor:
    return torch.isin(input_ids, placeholder_tensor).unsqueeze(-1)


def get_embedding_and_mask(
    data_embedding_func: Callable[[List[ImageDataItem]], torch.Tensor],
    embedding_items: List[ImageDataItem],
    placeholder_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate multimodal embeddings and create a mask for identifying their positions in the input sequence.

    Args:
        data_embedding_func: Function that generates embeddings for multimodal items
        embedding_items: List of multimodal items to embed
        placeholder_tensor: Tensor containing token IDs that serve as placeholders for multimodal content
        input_ids: The input token IDs tensor
        items_size: Cumulative sizes of multimodal items per request
        prefix_length: Prefix lengths for each request
        extend_length: Sequence lengths for each request
        items_offset_list: List of offset ranges for multimodal items in each request

    Returns:
        A tuple containing:
        - The generated embeddings tensor
        - A boolean mask tensor indicating where these embeddings should be placed
    """
    # 1. Get embedding
    embedding = _get_precomputed_embedding(embedding_items)
    if embedding is None:
        embedding = _get_chunked_prefill_embedding(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
        )
        if embedding is None:
            return None, None
    # 2. Get mask

    special_multimodal_mask = _get_multimodal_mask(input_ids, placeholder_tensor)
    # 3. Adjust embedding length if needed
    # NOTE seems to be relavant to chunked prefill, need further investigation
    # embedding = _adjust_embedding_length(embedding, special_multimodal_mask, logger)
    return embedding, special_multimodal_mask


def embed_mm_inputs(
    image_inputs_list: List[ImageInputs],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    input_embedding: nn.Embedding,
    multimodal_model: nn.Module = None,
) -> Optional[torch.Tensor]:
    """
    Embed multimodal inputs and integrate them with text token embeddings.

    Args:
        mm_inputs_list: List of multimodal inputs to process
        extend_prefix_lens: Prefix lengths for each request
        extend_seq_lens: Sequence lengths for each request
        input_ids: Input token IDs tensor
        input_embedding: Embedding layer for text tokens
        placeholder_tokens: Token IDs for multimodal placeholders (uses pad_values if None)

    Returns:
        Combined embedding tensor with multimodal content integrated
    """
    if image_inputs_list is None:
        return None

    # 1. Calculate the multimodal data which exists in input_ids, with the help of pad_values
    # we assume that multimodal data are represented with its pad_values in input_ids
    items = []
    for image_inputs in image_inputs_list:
        items += [item for item in image_inputs.image_items if item is not None]
    
    embedder = getattr(multimodal_model, f"get_image_feature", None)
    if len(items) != 0:
        assert embedder is not None, f"no embedding method found"
        placeholder_tensor = torch.as_tensor(
            [item.pad_value for item in items],
            device=input_ids.device,
        )
        # calculate per request items length offset
        items_size = torch.zeros(len(image_inputs_list) + 1, dtype=int)
        items_offsets = []
        for i, image_inputs in enumerate(image_inputs_list):
            image_items = [
                item
                for item in image_inputs.image_items
            ]
            items_size[i + 1] = len(image_items)
            items_offsets.append(
                flatten_nested_list([item.offsets for item in image_items])
            )
        items_size = torch.cumsum(items_size, dim=0).tolist()

        embedding, mask = get_embedding_and_mask(
            data_embedding_func=embedder,
            embedding_items=items,
            placeholder_tensor=placeholder_tensor,
            input_ids=input_ids,
            items_size=items_size,
            prefix_length=extend_prefix_lens,
            extend_length=extend_seq_lens,
            items_offset_list=items_offsets,
        )


    # 3. Get input embeddings
    vocab_size = input_embedding.num_embeddings
    # Important: clamp after getting original multimodal regions
    # Clamp input ids. This is because the input_ids for the multimodal tokens are
    # filled with the hash values of the multimodal for the prefix matching in the radix attention.
    # There values are useless because their embeddings will be replaced by vision embeddings anyway.
    input_ids.clamp_(min=0, max=vocab_size - 1)

    inputs_embeds = input_embedding(input_ids)
    
    # 4. scatter embeddings into input embedding
    indices = torch.where(mask.squeeze(dim=-1))[0]
    inputs_embeds[indices] = embedding.to(inputs_embeds.device, inputs_embeds.dtype)
    
    return inputs_embeds


def general_mm_embed_routine(
    input_ids: torch.Tensor,
    image_inputs: List[ImageDataItem],
    language_model: nn.Module,
    multimodal_model: Optional[nn.Module] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Process multimodal inputs and forward through language model.

    Args:
        input_ids: Input token IDs tensor
        forward_batch: Batch information for model forward pass
        language_model: Base language model to use
        data_embedding_funcs: A dictionary mapping from modality type to the corresponding embedding function.
        placeholder_tokens: Token IDs for multimodal placeholders
        use_deepstack: Whether to use deepstack embeddings for each modality, default False
        **kwargs: Additional arguments passed to language model

    Returns:
        Hidden states from language model forward pass
    """
    
    
    assert hasattr(language_model, "get_input_embeddings")
    embed_tokens = language_model.get_input_embeddings()
    if (
        image_inputs is not None # meaning this is the decoding phase
    ):
        context = get_context()
        extend_prefix_lens = context.extend_prefix_lens
        extend_seq_lens = context.extend_seq_lens
        
        image_inputs_list = [
            image_input for image_input in image_inputs if image_input is not None
        ]
        extend_prefix_lens = [
            prefix_len
            for i, prefix_len in enumerate(extend_prefix_lens)
            if image_inputs[i] is not None
        ]
        extend_seq_lens = [
            seq_len
            for i, seq_len in enumerate(extend_seq_lens)
            if image_inputs[i] is not None
        ]
        inputs_embeds = embed_mm_inputs(
            image_inputs_list=image_inputs_list,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            input_ids=input_ids,
            multimodal_model=multimodal_model,
            input_embedding=embed_tokens,
        )
        # I guess this is to free memory in the context
        set_context_field("image_inputs", None)
    else:
        inputs_embeds = embed_tokens(input_ids)

    hidden_states = language_model(
        input_ids=None,
        input_embeds=inputs_embeds,
        **kwargs,
    )
    return hidden_states

