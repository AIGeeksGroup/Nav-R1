import json
import os
import time
import base64
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any

import openai
from pathlib import Path

def encode_image(image_path: str) -> Optional[str]:
    """Encode an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Warning: Image not found: {image_path}")
        return None

@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str = "gemini-2.5-pro"
    max_tokens: int = 4096
    temperature: float = 0.7
    request_timeout_s: int = 120
    max_retries: int = 3
    retry_base_delay_s: float = 1.5


def _init_openai_client(cfg: LLMConfig) -> None:
    """Initialize OpenAI client settings for the current process."""
    openai.api_key = cfg.api_key
    openai.base_url = cfg.base_url


def query_LLM_with_images(
    instruction: str,
    images: List[str],
    cfg: Optional[LLMConfig] = None,
    max_images: Optional[int] = None,
) -> str:
    """
    Call the specified multimodal model to generate scene-understanding reasoning
    based only on the instruction and images.
    """
    # Resolve configuration from args/env with safe defaults
    if cfg is None:
        cfg = LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", ""),
            model=os.getenv("OPENAI_MODEL", "gemini-2.5-pro"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            request_timeout_s=int(os.getenv("OPENAI_TIMEOUT_S", "120")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
            retry_base_delay_s=float(os.getenv("OPENAI_RETRY_BASE_DELAY_S", "1.5")),
        )

    _init_openai_client(cfg)

    # Build text prompt - only includes the instruction
    text_prompt = f"""You are an expert in visual navigation and spatial reasoning.

Navigation Instruction: {instruction}

You are given multiple viewpoint images showing different perspectives along a trajectory path (5 images per viewpoint location).

Please provide detailed reasoning about the scene and how to navigate to complete this instruction:

1. **Scene Understanding**: Analyze what you observe in the images - describe the rooms, objects, furniture, walls, doorways, and overall spatial layout in detail.

2. **Instruction Analysis**: Break down what the navigation instruction requires - what are the key landmarks, directions, or targets mentioned?

3. **Navigation Strategy**: Based on the visual information from the images, explain step-by-step how you would navigate through this space to accomplish the instruction. What visual cues would guide each movement decision?

4. **Spatial Reasoning**: Describe the spatial relationships between different areas shown in the images and how they connect to form a path.

Provide comprehensive reasoning that connects what you see in the images to how you would navigate."""

    # Prepare message content (includes text and images)
    content: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt}]

    # Limit and add images
    selected_images = images[:max_images] if max_images else images
    for img_path in selected_images:
        base64_image = encode_image(img_path)
        if not base64_image:
            continue
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpg;base64,{base64_image}"},
        })

    # Call with retries and basic backoff
    last_error: Optional[Exception] = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            response = openai.chat.completions.create(
                model=cfg.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant specialized in visual navigation. Analyze the given images and provide detailed reasoning about scene understanding and navigation strategy.",
                    },
                    {"role": "user", "content": content},
                ],
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                timeout=cfg.request_timeout_s,
            )

            msg = response.choices[0].message
            reasoning = getattr(msg, "reasoning_content", "").strip()
            answer = msg.content.strip()

            # Merge reasoning_content and content as the <think> section
            think_content = f"{reasoning}\n\n{answer}" if reasoning else answer
            print(f"✓ Reasoning generated for instruction: {instruction[:50]}...")
            return think_content

        except Exception as e:
            last_error = e
            if attempt < cfg.max_retries:
                sleep_s = cfg.retry_base_delay_s * (2 ** (attempt - 1))
                print(f"Retry {attempt}/{cfg.max_retries} after error: {e}. Sleeping {sleep_s:.1f}s...")
                time.sleep(sleep_s)
            else:
                break

    error_msg = f"Error calling LLM after {cfg.max_retries} attempts: {last_error}"
    print(error_msg)
    return error_msg

def process_episode(
    episode: Dict[str, Any],
    cfg: Optional[LLMConfig] = None,
    max_images: Optional[int] = None,
) -> str:
    """Process a single episode."""
    instruction = episode.get("instruction", {}).get("instruction_text", "")
    images = episode.get("images", [])
    ground_truth_actions = episode.get("navigation_actions", [])
    
    print(f"\nProcessing Episode {episode['episode_id']}:")
    print(f"  Instruction: {instruction}")
    print(f"  Images: {len(images)} images")
    
    # Get model-generated reasoning (based only on instruction and images)
    think_content = query_LLM_with_images(
        instruction=instruction,
        images=images,
        cfg=cfg,
        max_images=max_images,
    )
    
    # Build full CoT format: <think>model reasoning</think><answer>ground-truth actions</answer>
    cot_result = f"<think>\n{think_content}\n</think>\n\n<answer>\n{json.dumps(ground_truth_actions, indent=2)}\n</answer>"
    
    return cot_result

def process_json_file(
    json_path: str,
    output_path: str,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    save_every: int = 5,
    reuse_every_n: int = 3,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    """Read the JSON file and process all episodes."""
    
    print(f"Reading JSON file: {json_path}")
    
    # Read JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_episodes = len(data["episodes"])
    print(f"Found {total_episodes} episodes to process\n")
    
    results: List[Dict[str, Any]] = []
    
    # Process each episode
    cot_result: Optional[str] = None
    cfg = LLMConfig(
        api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
        base_url=base_url or os.getenv("OPENAI_BASE_URL", ""),
        model=model or os.getenv("OPENAI_MODEL", "gemini-2.5-pro"),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        request_timeout_s=int(os.getenv("OPENAI_TIMEOUT_S", "120")),
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
        retry_base_delay_s=float(os.getenv("OPENAI_RETRY_BASE_DELAY_S", "1.5")),
    )
    for i, episode in enumerate(data["episodes"], 1):
        
        if reuse_every_n <= 1 or i % reuse_every_n == 1:
            
            print(f"\n{'='*60}")
            print(f"Processing episode {i}/{total_episodes}")
            print(f"Episode ID: {episode['episode_id']}, Trajectory ID: {episode['trajectory_id']}")
            print(f"{'='*60}")
            
            cot_result = process_episode(episode, cfg=cfg, max_images=max_images)
            
            # Save result
            episode_result: Dict[str, Any] = {
                "episode_id": episode.get("episode_id"),
                "trajectory_id": episode.get("trajectory_id"),
                "scene_id": episode.get("scene_id"),
                "instruction": episode.get("instruction", {}).get("instruction_text"),
                "start_position": episode.get("start_position"),
                "goal_position": (episode.get("goals", [{}])[0] or {}).get("position"),
                "geodesic_distance": (episode.get("info", {}) or {}).get("geodesic_distance"),
                "num_images": len(episode.get("images", [])),
                "images": episode.get("images", []),
                "cot_data": cot_result  # Complete <think>...</think><answer>...</answer>
            }
            # Include RxR-specific reference_viewpoints if present
            if "reference_viewpoints" in episode:
                episode_result["reference_viewpoints"] = episode["reference_viewpoints"]
            results.append(episode_result)
        
            
        else:
            print(f"\n{'='*60}")
            print(f"Processing episode {i}/{total_episodes}")
            print(f"Episode ID: {episode['episode_id']}, Trajectory ID: {episode['trajectory_id']}")
            print(f"{'='*60}")
            episode_result = {
                "episode_id": episode.get("episode_id"),
                "trajectory_id": episode.get("trajectory_id"),
                "scene_id": episode.get("scene_id"),
                "instruction": episode.get("instruction", {}).get("instruction_text"),
                "start_position": episode.get("start_position"),
                "goal_position": (episode.get("goals", [{}])[0] or {}).get("position"),
                "geodesic_distance": (episode.get("info", {}) or {}).get("geodesic_distance"),
                "num_images": len(episode.get("images", [])),
                "cot_data": cot_result
            }
            if "reference_viewpoints" in episode:
                episode_result["reference_viewpoints"] = episode["reference_viewpoints"]
            results.append(episode_result)

        print(f"✓ Episode {episode['episode_id']} processed successfully")

        # Save every N episodes (prevent data loss)
        if save_every > 0 and i % save_every == 0:
            temp_output = output_path.replace('.json', f'_temp_{i}.json')
            with open(temp_output, 'w', encoding='utf-8') as f:
                json.dump({"total_episodes": len(results), "results": results}, 
                         f, indent=2, ensure_ascii=False)
            print(f"  → Intermediate results saved to {temp_output}")
    
    # Save final results
    output_data: Dict[str, Any] = {
        "total_episodes": len(results),
        "processed_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "model": cfg.model,
        "results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ All episodes processed successfully!")
    print(f"✓ Results saved to: {output_path}")
    print(f"✓ Total episodes: {len(results)}")
    print(f"{'='*60}\n")
    
    return output_data

# Usage example / CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build CoT data from annotated navigation episodes (R2R/RxR).")
    parser.add_argument("--input", default="train_annotated.json", help="Input JSON file path")
    parser.add_argument("--output", default="cot_results.json", help="Output JSON file path")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gemini-2.5-pro"), help="Model name")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""), dest="base_url", help="API base URL")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), dest="api_key", help="API key (or set OPENAI_API_KEY)")
    parser.add_argument("--save-every", type=int, default=5, help="Save temp results every N episodes")
    parser.add_argument("--reuse-every-n", type=int, default=3, help="Call LLM every N episodes and reuse result in between")
    parser.add_argument("--max-images", type=int, default=None, help="Max number of images to include per episode")
    parser.add_argument("--dataset", choices=["r2r", "rxr"], default=None, help="Optional dataset tag for output metadata")

    args = parser.parse_args()

    json_file_path = args.input
    output_file_path = args.output

    # Ensure the input file exists
    if not Path(json_file_path).exists():
        print(f"Error: Input file not found: {json_file_path}")
        print("Please update the --input argument with the correct path.")
        exit(1)

    # Process file
    process_json_file(
        json_file_path,
        output_file_path,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        save_every=args.save_every,
        reuse_every_n=args.reuse_every_n,
        max_images=args.max_images,
    )