# RegisterBridgeMM Work Plan

## Goal

Implement a new `RegisterBridgeMM` family inside this codebase without modifying existing original files.
The new family will reuse the reference project's mature multimodal training, validation, predictor,
results, dataset, and router stack, while replacing the front-end feature extractor with our
`DualDINOv2 + RegisterBridge` pipeline.

## Design Principles

- Do not edit existing original family files such as `YOLOMM`, `RTDETRMM`, `ultralytics/nn/tasks.py`, or
  current multimodal dataset/build/predictor implementations.
- Reuse the reference project's existing engineering stack whenever possible:
  - `ultralytics.data.build.build_yolo_dataset`
  - `ultralytics.data.dataset.YOLOMultiModalImageDataset`
  - `ultralytics.data.multimodal_augment`
  - `ultralytics.engine.multimodal.predictor`
  - `ultralytics.engine.multimodal.results`
  - existing YOLO detect loss/validator/results pipeline
- Add a new model family as a separate namespace.
- Phase 1 targets YOLO detect only. RT-DETR integration is Phase 2.

## Reference Architecture Findings

The reference project's multimodal extension is not just new modules. It is a 5-layer architecture:

1. family entry + `task_map` dispatch
2. YAML parse + `mm_router` injection
3. shared multimodal dataset/build/augment pipeline
4. task-specific trainer/validator/predictor adapters
5. multimodal inference/results objects

Therefore `RegisterBridgeMM` must provide all of the following:

- a `Model` family entry class
- task-specific trainer/validator/predictor adapters
- a dedicated model-construction path
- compatibility with the existing multimodal data/router stack

## Phase 1 Scope: RegisterBridgeMM-YOLO

### New Files

#### Family entry
- `ultralytics/models/registerbridgemm/__init__.py`
- `ultralytics/models/registerbridgemm/model.py`

#### Task adapters
- `ultralytics/models/registerbridgemm/train.py`
- `ultralytics/models/registerbridgemm/val.py`
- `ultralytics/models/registerbridgemm/predict.py`

#### Model construction
- `ultralytics/nn/tasks_registerbridge.py`

#### Method modules
- `ultralytics/nn/modules/registerbridge/__init__.py`
- `ultralytics/nn/modules/registerbridge/backbone.py`
- `ultralytics/nn/modules/registerbridge/bridge.py`
- `ultralytics/nn/modules/registerbridge/neck.py`
- `ultralytics/nn/modules/registerbridge/model.py`

#### Configs
- `configs/registerbridgemm/registerbridge_yolo_dronevehicle.yaml`

## Phase 1 Call Flow

```text
RegisterBridgeMM(...)
  -> task_map['detect']
  -> RegisterBridgeMMTrainer / Validator / Predictor
  -> RegisterBridgeDetectionModel (from tasks_registerbridge.py)
  -> existing multimodal dataset/build/router path
  -> our DualDINOv2 + RegisterBridge feature extractor
  -> existing Ultralytics Detect head / loss / validator / results
```

## Why This Plan Is Safer

- keeps mature detect/loss/validator/results code unchanged
- avoids continuing to patch the current standalone DETR/YOLO implementations
- uses the proven multimodal engineering structure of this reference project
- keeps all new work isolated under `registerbridgemm`

## Immediate Next Steps

1. scaffold the new `registerbridgemm` family files
2. create a thin detect trainer/validator/predictor layer by subclassing existing multimodal YOLO adapters
3. add a dedicated `RegisterBridgeDetectionModel` build path
4. migrate our method modules into `ultralytics/nn/modules/registerbridge/`
5. wire the new feature extractor into the existing YOLO detect head
6. run a model-build smoke test
7. run a 64-image overfit smoke test

## Phase 2

After YOLO detect is stable, add `RegisterBridgeMM-RTDETR` using the same family architecture,
but only after Phase 1 proves the feature extractor and data integration are healthy.
