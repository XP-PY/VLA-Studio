# Recording Profiles

Recording profiles select only direct source datasets. Derived annotation/training datasets are
never valid recording targets. The default profile appends one episode to the `place_one_vegetable`
source subset:

```bash
./scripts/record.sh
```

Select another task family without editing `env.sh` or `record.sh`:

```bash
RECORD_CONFIG=configs/recordings/single_table_bus_table.env ./scripts/record.sh
```

Override a prompt and the number of new episodes for one invocation:

```bash
RECORD_CONFIG=configs/recordings/single_table_place_one_vegetable.env \
RECORD_TASK="Place the cucumber on the blue plate." \
RECORD_NUM_EPISODES=5 \
./scripts/record.sh
```

`RECORD_NUM_EPISODES` is the number of new episodes to record in that invocation, not the desired
total dataset size. Existing LeRobot data is resumed and appended. After each attempt, the passive
recorder still asks whether to save or discard it by default.

Use a dry-run to inspect the resolved output path and command without connecting hardware:

```bash
RECORD_DRY_RUN=true \
RECORD_CONFIG=configs/recordings/single_table_clear_specified_plate.env \
./scripts/record.sh
```

| Profile | Source subset |
|---|---|
| `single_table_place_one_vegetable.env` | Place one selected vegetable on one selected plate. |
| `single_table_arrange_two_vegetables.env` | Place two vegetables in the requested order. |
| `single_table_clear_specified_plate.env` | Move the vegetable from a specified plate to the collection box. |
| `single_table_bus_table.env` | Move all vegetables to the collection box. |
