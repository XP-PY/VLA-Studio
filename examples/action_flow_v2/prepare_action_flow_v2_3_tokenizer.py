#!/usr/bin/env python

"""Append AFV2.3 planning tokens to the existing AFV2.2 tokenizer artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoProcessor, AutoTokenizer

from lerobot.policies.action_flow_v2_3.configuration_action_flow_v2_3 import ActionFlowV23Config
from lerobot.policies.action_flow_v2_3.tokenization_action_flow_v2_3 import (
    ActionFlowV23TargetTokenizer,
    FAST_TOKENIZER_DIR,
    VLM_TOKENIZER_DIR,
)

DEFAULT_SOURCE = Path("artifacts/tokenizers/action_flow_v2_2_smolvlm_fast")
DEFAULT_OUTPUT = Path("artifacts/tokenizers/action_flow_v2_3_smolvlm_fast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest_path = args.source_artifact / "action_flow_v2_2_tokenizer.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"AFV2.2 tokenizer manifest not found: {source_manifest_path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output artifact exists: {args.output_dir}. Use --overwrite.")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    config = ActionFlowV23Config(
        device="cpu",
        chunk_size=args.chunk_size,
        n_action_steps=min(10, args.chunk_size),
    )
    vlm_tokenizer = AutoTokenizer.from_pretrained(
        args.source_artifact / VLM_TOKENIZER_DIR,
        local_files_only=True,
    )
    action_tokenizer = AutoProcessor.from_pretrained(
        args.source_artifact / FAST_TOKENIZER_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = ActionFlowV23TargetTokenizer(
        config=config,
        vlm_tokenizer=vlm_tokenizer,
        action_tokenizer=action_tokenizer,
        entity_names=source_manifest["entity_names"],
    )
    grasp = tokenizer.serialize_plan(stage="grasp", target_entity="tomato")
    complete = tokenizer.serialize_plan(stage="complete", target_entity="")
    assert tokenizer.deserialize_plan(grasp).stage == "grasp"
    assert tokenizer.deserialize_plan(complete).stage == "complete"
    tokenizer.save_pretrained(args.output_dir)
    reloaded = ActionFlowV23TargetTokenizer.from_pretrained(args.output_dir, config=config)
    assert reloaded.serialize_plan(stage="put", target_entity="pink plate") == (
        tokenizer.serialize_plan(stage="put", target_entity="pink plate")
    )
    validation = {
        "source_artifact_format": source_manifest["format"],
        "extended_vocabulary_size": len(tokenizer.vlm_tokenizer),
        "action_flow_v2_2_vocab_size": config.action_flow_v2_2_vocab_size,
        "planning_token_ids": config.token_contract()["planning_token_ids"],
        "entity_names": list(tokenizer.entity_names),
        "checkpoint_reload_exact": True,
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
