class TrainableSchedule:
    def __init__(self, spec):
        self.spec = (spec or "none").strip()
        self.enabled = self.spec.lower() not in ("", "none", "off", "false", "0")
        self.stages = self._parse(self.spec) if self.enabled else []
        self.current_index = None
        self.current_stats = None

    def _parse(self, spec):
        stages = []
        for raw_stage in spec.split(";"):
            raw_stage = raw_stage.strip()
            if not raw_stage:
                continue
            if ":" not in raw_stage:
                raise ValueError(
                    "Invalid trainable schedule stage '{}'. Expected '<step>:<tokens>'.".format(raw_stage)
                )
            raw_step, raw_tokens = raw_stage.split(":", 1)
            step = int(raw_step.strip())
            if step < 0:
                raise ValueError("Trainable schedule step must be non-negative: {}".format(raw_stage))
            tokens = tuple(
                token.strip().lower()
                for chunk in raw_tokens.split(",")
                for token in chunk.split("+")
                if token.strip()
            )
            if not tokens:
                raise ValueError("Trainable schedule stage has no trainable tokens: {}".format(raw_stage))
            unknown = sorted(set(tokens) - set(known_tokens()))
            if unknown:
                raise ValueError(
                    "Unknown trainable schedule token(s): {}. Known tokens: {}".format(
                        ", ".join(unknown), ", ".join(known_tokens())
                    )
                )
            stages.append({"step": step, "tokens": tokens, "label": raw_stage})
        if not stages:
            raise ValueError("Trainable schedule is enabled but contains no stages.")
        stages = sorted(stages, key=lambda item: item["step"])
        if stages[0]["step"] != 0:
            raise ValueError("Trainable schedule must start at step 0.")
        for left, right in zip(stages, stages[1:]):
            if left["step"] == right["step"]:
                raise ValueError("Duplicate trainable schedule step: {}".format(left["step"]))
        return stages

    def describe(self):
        if not self.enabled:
            return "none"
        return ";".join(stage["label"] for stage in self.stages)

    def stage_for_step(self, step):
        if not self.enabled:
            return None, None
        index = 0
        for candidate_index, stage in enumerate(self.stages):
            if step >= stage["step"]:
                index = candidate_index
            else:
                break
        return index, self.stages[index]

    def apply(self, model, step, force=False):
        if not self.enabled:
            self.current_stats = summarize_trainable(model, stage_index=-1, stage_label="all")
            return dict(self.current_stats, changed=False)

        stage_index, stage = self.stage_for_step(step)
        changed = force or stage_index != self.current_index
        if changed:
            apply_tokens(model, stage["tokens"])
            self.current_index = stage_index
        self.current_stats = summarize_trainable(
            model,
            stage_index=stage_index,
            stage_label=stage["label"],
            stage_step=stage["step"],
            tokens=stage["tokens"],
        )
        self.current_stats["changed"] = changed
        if self.current_stats["trainable_params"] <= 0:
            raise ValueError(
                "Trainable schedule stage '{}' selected zero parameters. "
                "Check enabled architecture flags and schedule tokens.".format(stage["label"])
            )
        return self.current_stats


def known_tokens():
    return (
        "all",
        "brfrc",
        "baseline",
        "lf_prior",
        "bottleneck",
        "fusion",
        "level3",
        "level2",
        "level1",
        "encoder",
        "decoder",
        "output",
        "deconv",
    )


def normalize_name(name):
    return name.replace("module.", "", 1) if name.startswith("module.") else name


def apply_tokens(model, tokens):
    tokens = set(tokens)
    for name, param in model.named_parameters():
        param.requires_grad = is_trainable_name(normalize_name(name), tokens)


def is_trainable_name(name, tokens):
    if "all" in tokens:
        return True
    return any(matches_token(name, token) for token in tokens)


def matches_token(name, token):
    if token == "brfrc":
        return name.startswith("corrector.")
    if token == "baseline":
        return name.startswith("baseline.")
    if name.startswith("baseline."):
        name = name[len("baseline."):]
    if token == "lf_prior":
        return name.startswith("lf_prior.")
    if token == "bottleneck":
        return (
            name.startswith("down3.")
            or name.startswith("fe_level_3.")
            or name.startswith("level3_block")
            or name.startswith("mix1.")
        )
    if token == "fusion":
        return name.startswith("mix1.") or name.startswith("mix2.")
    if token == "level3":
        return name.startswith("fe_level_3.") or name.startswith("level3_block")
    if token == "level2":
        return (
            name.startswith("fe_level_2.")
            or name.startswith("down_level2_block")
            or name.startswith("up_level2_block")
        )
    if token == "level1":
        return name.startswith("down_level1_block") or name.startswith("up_level1_block")
    if token == "encoder":
        return (
            name.startswith("down1.")
            or name.startswith("down2.")
            or name.startswith("down3.")
            or name.startswith("down_level")
            or name.startswith("fe_level_")
            or name.startswith("level3_block")
        )
    if token == "decoder":
        return (
            name.startswith("up1.")
            or name.startswith("up2.")
            or name.startswith("up3.")
            or name.startswith("up_level")
        )
    if token == "output":
        return name.startswith("up3.")
    if token == "deconv":
        return ".conv1.conv1_" in name
    return False


def summarize_trainable(model, stage_index=-1, stage_label="all", stage_step=0, tokens=None):
    total_params = 0
    trainable_params = 0
    total_tensors = 0
    trainable_tensors = 0
    trainable_by_scope = {}
    frozen_by_scope = {}

    for name, param in model.named_parameters():
        clean_name = normalize_name(name)
        scope = clean_name.split(".", 1)[0]
        count = param.numel()
        total_params += count
        total_tensors += 1
        if param.requires_grad:
            trainable_params += count
            trainable_tensors += 1
            trainable_by_scope[scope] = trainable_by_scope.get(scope, 0) + count
        else:
            frozen_by_scope[scope] = frozen_by_scope.get(scope, 0) + count

    return {
        "stage_index": stage_index,
        "stage_label": stage_label,
        "stage_step": stage_step,
        "tokens": list(tokens or ()),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "total_tensors": total_tensors,
        "trainable_tensors": trainable_tensors,
        "trainable_by_scope": trainable_by_scope,
        "frozen_by_scope": frozen_by_scope,
    }
