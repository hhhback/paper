from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    name: str
    constructor_type: str       # "no_memory" | "full_context" | "flat_memory" | "fixed_hierarchy" | "ours"
    disable_dynamic_routing: bool = False
    disable_elevation: bool = False
    disable_backward: bool = False
    implicit_only: bool = False
    explicit_only: bool = False
    context_budget: int = 8192  # max tokens for prediction context


PRESETS = {
    # Baselines
    "no_memory": ExperimentConfig(name="No-Memory", constructor_type="no_memory"),
    "full_context": ExperimentConfig(name="Full-Context", constructor_type="full_context"),
    "flat_memory": ExperimentConfig(name="Flat-Memory", constructor_type="flat_memory"),
    "fixed_hierarchy": ExperimentConfig(name="Fixed-Hierarchy", constructor_type="fixed_hierarchy", disable_dynamic_routing=True, disable_backward=True),
    "ours": ExperimentConfig(name="Ours", constructor_type="ours"),
    # Ablations (all use constructor_type="ours" with flags)
    "ablation_dynamic": ExperimentConfig(name="-dynamic", constructor_type="ours", disable_dynamic_routing=True),
    "ablation_elevation": ExperimentConfig(name="-elevation", constructor_type="ours", disable_elevation=True),
    "ablation_backward": ExperimentConfig(name="-backward", constructor_type="ours", disable_backward=True),
    "ablation_no_implicit": ExperimentConfig(name="w/o implicit", constructor_type="ours", explicit_only=True),
    "ablation_no_explicit": ExperimentConfig(name="w/o explicit", constructor_type="ours", implicit_only=True),
}

BASELINE_NAMES = ["no_memory", "full_context", "flat_memory", "fixed_hierarchy", "ours"]

ABLATION_NAMES = [
    "ablation_dynamic",
    "ablation_elevation",
    "ablation_backward",
    "ablation_no_implicit",
    "ablation_no_explicit",
]

POI_CATEGORIES = [
    "음식점",
    "카페/디저트",
    "교통/모빌리티",
    "여행/숙박",
    "레저/스포츠",
    "문화/예술",
    "자기계발",
    "반려동물",
    "쇼핑",
    "의료",
    "미용",
]


def get_preset(name: str) -> ExperimentConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
