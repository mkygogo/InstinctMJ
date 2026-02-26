from __future__ import annotations

import numpy as np
import os
import torch
from typing import TYPE_CHECKING

import onnxruntime as ort

if TYPE_CHECKING:
    from typing import Callable


def _build_ort_providers() -> list[str]:
    """Prefer CUDA/CPU providers and skip TensorRT probing noise."""
    available = set(ort.get_available_providers())
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    # Fallback to whatever ORT reports if neither CUDA nor CPU is visible.
    return providers if providers else list(ort.get_available_providers())


def _run_onnx_with_batch_support(
    session: ort.InferenceSession, input_name: str, batched_input: np.ndarray
) -> np.ndarray:
    """Run ONNX with fallback for models exported with fixed batch size = 1."""
    expected_batch = session.get_inputs()[0].shape[0]
    if isinstance(expected_batch, int) and expected_batch > 0 and batched_input.shape[0] != expected_batch:
        if expected_batch != 1:
            raise ValueError(
                f"ONNX model expects fixed batch={expected_batch}, but got batch={batched_input.shape[0]}."
            )
        outputs = [session.run(None, {input_name: batched_input[i : i + 1]})[0] for i in range(batched_input.shape[0])]
        return np.concatenate(outputs, axis=0)
    return session.run(None, {input_name: batched_input})[0]


def load_parkour_onnx_model(
    model_dir: str, get_subobs_func: Callable, depth_shape: tuple, proprio_slice: slice
) -> Callable:
    """Load the ONNX model as policy, but only for parkour task setting."""
    ort_providers = _build_ort_providers()
    encoder = ort.InferenceSession(os.path.join(model_dir, "0-depth_encoder.onnx"), providers=ort_providers)
    actor = ort.InferenceSession(os.path.join(model_dir, "actor.onnx"), providers=ort_providers)
    encoder_input_name = encoder.get_inputs()[0].name
    actor_input_name = actor.get_inputs()[0].name

    def policy(obs: torch.Tensor) -> torch.Tensor:
        depth_image_input = get_subobs_func(obs)
        depth_image_input = depth_image_input.cpu().numpy()
        depth_image_input = depth_image_input.reshape((-1, *depth_shape))
        depth_image_output = _run_onnx_with_batch_support(encoder, encoder_input_name, depth_image_input)
        actor_input = np.concatenate(
            [
                obs.cpu().numpy()[:, proprio_slice],
                depth_image_output,
            ],
            axis=1,
        )
        actor_output = _run_onnx_with_batch_support(actor, actor_input_name, actor_input)
        return torch.from_numpy(actor_output).to(obs.device)

    return policy
