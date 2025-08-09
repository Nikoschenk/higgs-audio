"""Example script for voice cloning and TTS generation using HiggsAudio."""

import click
import soundfile as sf
import os
import torch
import pickle
from loguru import logger
from typing import Optional

from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import StaticCache
from dataclasses import asdict


CURR_DIR = os.path.dirname(os.path.abspath(__file__))


def clone_voice(
    ref_audio_path: str,
    ref_text_path: str,
    output_clone_path: str,
    audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer",
    device: str = "auto"
) -> None:
    """Clone a voice from reference audio and save the voice profile to disk.
    
    Args:
        ref_audio_path: Path to reference audio file
        ref_text_path: Path to text file containing the transcript of reference audio
        output_clone_path: Path to save the cloned voice profile
        audio_tokenizer_path: Path to audio tokenizer
        device: Device to use for processing
    """
    logger.info(f"Starting voice cloning process...")
    logger.info(f"Reference audio: {ref_audio_path}")
    logger.info(f"Reference text: {ref_text_path}")
    
    # Set up device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # For MPS, use CPU for audio tokenizer due to embedding operation limitations
    audio_tokenizer_device = "cpu" if device == "mps" else device
    
    # Load audio tokenizer
    logger.info(f"Loading audio tokenizer on device: {audio_tokenizer_device}")
    audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_path, device=audio_tokenizer_device)
    
    # Verify files exist
    if not os.path.exists(ref_audio_path):
        raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")
    if not os.path.exists(ref_text_path):
        raise FileNotFoundError(f"Reference text file not found: {ref_text_path}")
    
    # Read reference text
    with open(ref_text_path, "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
    
    # Encode reference audio
    logger.info("Encoding reference audio...")
    audio_tokens = audio_tokenizer.encode(ref_audio_path)
    
    # Create voice profile data structure
    voice_profile = {
        "audio_tokens": audio_tokens,
        "reference_text": ref_text,
        "reference_audio_path": ref_audio_path,
        "audio_tokenizer_path": audio_tokenizer_path,
        "device_used": device
    }
    
    # Save voice profile
    os.makedirs(os.path.dirname(output_clone_path), exist_ok=True)
    with open(output_clone_path, "wb") as f:
        pickle.dump(voice_profile, f)
    
    logger.info(f"Voice cloning completed. Profile saved to: {output_clone_path}")


def speak_with_cloned_voice(
    cloned_voice_path: str,
    text_input: str,
    output_audio_path: str,
    model_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
    scene_prompt: Optional[str] = None,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    max_new_tokens: int = 2048,
    device: str = "auto"
) -> None:
    """Generate TTS audio using a cloned voice profile.
    
    Args:
        cloned_voice_path: Path to saved voice profile
        text_input: Text to convert to speech
        output_audio_path: Path to save generated audio
        model_path: Path to HiggsAudio model
        scene_prompt: Optional scene description
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        max_new_tokens: Maximum tokens to generate
        device: Device to use for generation
    """
    logger.info(f"Starting TTS generation with cloned voice...")
    logger.info(f"Cloned voice profile: {cloned_voice_path}")
    logger.info(f"Text input: {text_input}")
    
    # Set up device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Load voice profile
    if not os.path.exists(cloned_voice_path):
        raise FileNotFoundError(f"Cloned voice profile not found: {cloned_voice_path}")
    
    with open(cloned_voice_path, "rb") as f:
        voice_profile = pickle.load(f)
    
    logger.info("Voice profile loaded successfully")
    
    # Load audio tokenizer
    audio_tokenizer_device = "cpu" if device == "mps" else device
    audio_tokenizer = load_higgs_audio_tokenizer(
        voice_profile["audio_tokenizer_path"], 
        device=audio_tokenizer_device
    )
    
    # Load model
    logger.info(f"Loading model on device: {device}")
    model = HiggsAudioModel.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    
    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    
    # Set up collator
    collator = HiggsAudioSampleCollator(
        whisper_processor=None,
        audio_in_token_id=config.audio_in_token_idx,
        audio_out_token_id=config.audio_out_token_idx,
        audio_stream_bos_id=config.audio_stream_bos_id,
        audio_stream_eos_id=config.audio_stream_eos_id,
        encode_whisper_embed=config.encode_whisper_embed,
        pad_token_id=config.pad_token_id,
        return_audio_in_tokens=config.encode_audio_in_tokens,
        use_delay_pattern=config.use_delay_pattern,
        round_to=1,
        audio_num_codebooks=config.audio_num_codebooks,
    )
    
    # Prepare messages
    messages = []
    
    # Add system message if scene prompt provided
    if scene_prompt:
        messages.append(Message(
            role="system",
            content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>"
        ))
    else:
        messages.append(Message(
            role="system",
            content="Generate audio following instruction."
        ))
    
    # Add reference audio as context
    messages.append(Message(
        role="user",
        content=voice_profile["reference_text"]
    ))
    messages.append(Message(
        role="assistant",
        content=AudioContent(audio_url="")
    ))
    
    # Add current generation request
    messages.append(Message(
        role="user",
        content=text_input
    ))
    
    # Prepare input
    chatml_sample = ChatMLSample(messages=messages)
    input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, tokenizer)
    postfix = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>\n\n", 
        add_special_tokens=False
    )
    input_tokens.extend(postfix)
    
    # Prepare dataset sample
    audio_ids = [voice_profile["audio_tokens"]]
    curr_sample = ChatMLDatasetSample(
        input_ids=torch.LongTensor(input_tokens),
        label_ids=None,
        audio_ids_concat=torch.concat([ele.cpu() for ele in audio_ids], dim=1),
        audio_ids_start=torch.cumsum(
            torch.tensor([0] + [ele.shape[1] for ele in audio_ids], dtype=torch.long), dim=0
        ),
        audio_waveforms_concat=None,
        audio_waveforms_start=None,
        audio_sample_rate=None,
        audio_speaker_indices=None,
    )
    
    # Prepare batch
    batch_data = collator([curr_sample])
    batch = asdict(batch_data)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.contiguous().to(device)
    
    # Generate audio
    logger.info("Generating audio...")
    with torch.inference_mode():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            tokenizer=tokenizer,
        )
    
    # Process audio output
    audio_out_ids_l = []
    for ele in outputs[1]:
        audio_out_ids = ele
        if config.use_delay_pattern:
            audio_out_ids = revert_delay_pattern(audio_out_ids)
        audio_out_ids_l.append(audio_out_ids.clip(0, audio_tokenizer.codebook_size - 1)[:, 1:-1])
    
    concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)
    
    # Fix MPS compatibility
    if concat_audio_out_ids.device.type == "mps":
        concat_audio_out_ids_cpu = concat_audio_out_ids.detach().cpu()
    else:
        concat_audio_out_ids_cpu = concat_audio_out_ids
    
    # Decode audio
    concat_wv = audio_tokenizer.decode(concat_audio_out_ids_cpu.unsqueeze(0))[0, 0]
    
    # Save output
    sr = 24000
    sf.write(output_audio_path, concat_wv, sr)
    logger.info(f"Generated audio saved to: {output_audio_path}")


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["clone", "speak", "both"]),
    default="both",
    help="Mode: 'clone' for voice cloning only, 'speak' for TTS only, 'both' for clone then speak"
)
@click.option(
    "--ref_audio",
    type=str,
    required=True,
    help="Path to reference audio file for cloning (required for clone and both modes)"
)
@click.option(
    "--ref_text",
    type=str,
    required=True,
    help="Path to reference text file (required for clone and both modes)"
)
@click.option(
    "--cloned_voice_path",
    type=str,
    default="cloned_voice.pkl",
    help="Path to save/load cloned voice profile"
)
@click.option(
    "--text_input",
    type=str,
    default="Hello, this is a test of the cloned voice.",
    help="Text to convert to speech (required for speak and both modes)"
)
@click.option(
    "--output_audio",
    type=str,
    default="generated_speech.wav",
    help="Path to save generated audio"
)
@click.option(
    "--model_path",
    type=str,
    default="bosonai/higgs-audio-v2-generation-3B-base",
    help="Path to HiggsAudio model"
)
@click.option(
    "--audio_tokenizer",
    type=str,
    default="bosonai/higgs-audio-v2-tokenizer",
    help="Path to audio tokenizer"
)
@click.option(
    "--scene_prompt",
    type=str,
    default=None,
    help="Optional scene description for generation context"
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="Sampling temperature"
)
@click.option(
    "--top_k",
    type=int,
    default=50,
    help="Top-k sampling parameter"
)
@click.option(
    "--top_p",
    type=float,
    default=0.95,
    help="Top-p sampling parameter"
)
@click.option(
    "--max_new_tokens",
    type=int,
    default=2048,
    help="Maximum tokens to generate"
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="auto",
    help="Device to use for processing"
)
def main(
    mode,
    ref_audio,
    ref_text,
    cloned_voice_path,
    text_input,
    output_audio,
    model_path,
    audio_tokenizer,
    scene_prompt,
    temperature,
    top_k,
    top_p,
    max_new_tokens,
    device
):
    """Voice cloning and TTS generation using HiggsAudio."""
    
    logger.info(f"Starting in {mode} mode")
    
    if mode in ["clone", "both"]:
        logger.info("=== Voice Cloning Phase ===")
        clone_voice(
            ref_audio_path=ref_audio,
            ref_text_path=ref_text,
            output_clone_path=cloned_voice_path,
            audio_tokenizer_path=audio_tokenizer,
            device=device
        )
    
    if mode in ["speak", "both"]:
        logger.info("=== TTS Generation Phase ===")
        speak_with_cloned_voice(
            cloned_voice_path=cloned_voice_path,
            text_input=text_input,
            output_audio_path=output_audio,
            model_path=model_path,
            scene_prompt=scene_prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            device=device
        )
    
    logger.info("Process completed successfully!")


if __name__ == "__main__":
    main()