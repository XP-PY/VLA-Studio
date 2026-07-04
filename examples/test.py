from lerobot.configs import parser
from dataclasses import dataclass
from lerobot.configs import (
    PreTrainedConfig
)
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

@dataclass
class TestConfig():
    policy: PreTrainedConfig | None = None
    test_name: str = 'TEST'

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]

    def validate(self) -> None:
        print(f"[INFO] policy: {self.policy}")
        print(f"[INFO] policy type: {self.policy.type}")
        print(f"[INFO] policy device: {self.policy.device}")


@parser.wrap()
def test(cfg: TestConfig):
    cfg.validate()

def main():
    test()


if __name__ == "__main__":
    main()
