# ABOUTME: SFT dataset from allenai/tulu-3-sft-mixture, filtered to single-turn.
# ABOUTME: Keeps single-turn rows, prepends the IF system message, and marks Persona IF provenance.

from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING


PERSONA_IF_SOURCE = "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980"
PERSONA_IF_COLUMN = "is_persona_if"


def _is_single_turn(messages):
    return (
        len(messages) == 2
        and messages[0]["role"] == "user"
        and messages[1]["role"] == "assistant"
    )


class TuluMixSFT(HFDataset):
    def __init__(self):
        super().__init__("tulumix")

    def _format_rows(self, dataset):
        single_turn = dataset.filter(
            lambda rows: [_is_single_turn(m) for m in rows["messages"]],
            batched=True,
        )

        def transform(row):
            prompt = row["messages"][0]["content"]
            response = row["messages"][1]["content"]
            return {
                "prompt": prompt,
                PERSONA_IF_COLUMN: row["source"] == PERSONA_IF_SOURCE,
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
            }

        return single_turn.map(transform, remove_columns=single_turn.column_names)

    def format_dataset(self):
        formatted = self._format_rows(self._dataset)
        formatted_dataset = formatted.train_test_split(
            test_size=min(200, len(formatted) - 1), shuffle=False
        )
        print(f"TuluMix SFT Dataset - {formatted_dataset}")
        print(f"Example row - {formatted_dataset['train'][0]}")
        self._dataset = formatted_dataset


if __name__ == "__main__":
    tulumix = TuluMixSFT()
    tulumix.load_from_huggingface("allenai/tulu-3-sft-mixture", split="train")
    tulumix.clear_old_datasets(prefix="sft-tulumix")
    tulumix.format_dataset()
    tulumix.save_dataset_to_disk(save_name="sft-tulumix")
