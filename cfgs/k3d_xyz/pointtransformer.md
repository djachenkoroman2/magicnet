# pointtransformer.yaml

Файл [pointtransformer.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/pointtransformer.yaml#L1) загружается вместе с [default.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/default.yaml#L4) через `cfg.load(..., recursive=True)`. То есть он не дублирует оптимизатор, scheduler и общие параметры датасета, а переопределяет только то, что реально нужно для `PointTransformer`.

**Что задаёт этот файл**
- `model` — сегментационный PointTransformer backbone/head в виде одного модуля `PTSeg`.
- `feature_keys` — какие признаки подаются на вход модели.
- `batch_size`, `val_batch_size`, `step_per_update` — более осторожные настройки под память.
- `dataloader.collate_fn` — специальный collate для variable-length point batches.
- `dataset.common.variable` — включает “плоский” режим данных с `pos: [N, 3]`, `x: [N, C]`, `o: [B]`.
- `dataset.train.voxel_max` — ограничение числа точек в train-crop, чтобы PointTransformer был практичным на K3DXYZ.

**Параметры `model`**
- `model.NAME: PTSeg` — сегментационная модель из [pointtransformer.py](/home/researcher/dev/PointNeXt/openpoints/models/backbone/pointtransformer.py#L218), где encoder, decoder и классификатор собраны в одном классе.
- `block: PointTransformerBlock` — базовый residual attention block Point Transformer V1.
- `blocks: [2, 3, 4, 6, 3]` — глубина по стадиям энкодера; это уже полноценный, а не “toy” вариант.
- `width: 32` — базовая ширина каналов. Дальше сеть расширяется по стадиям как `32, 64, 128, 256, 512`.
- `nsample: [8, 16, 16, 16, 16]` — число соседей для локального внимания/агрегации по стадиям.
- `in_channels: 4` — входные каналы `xyz + height`, согласованные с `feature_keys: pos,heights`.
- `num_classes: 8` — 8 логитов на точку для K3DXYZ.
- `dec_local_aggr: True` — после апсемплинга декодер тоже использует локальную агрегацию.
- `mid_res: False` — residual connection остаётся в стандартном варианте блока.

**Почему здесь нужен special collate**
- Обычные конфиги сегментации в этом репозитории работают с батчами вида `[B, N, C]`.
- `PointTransformer` здесь реализован в “variable-length” стиле: вход идёт как один плоский список точек `[N, C]` плюс `o`, где `o` — offset конца каждого объекта в батче.
- Поэтому в конфиге нужен `dataloader.collate_fn: concat_collate_fn`.
- И поэтому же включён `dataset.common.variable: True`.

**Практический смысл настроек**
- `batch_size: 1` и `step_per_update: 2` — безопасный старт по памяти; эффективный batch ближе к 2 mini-batch.
- `dataset.train.voxel_max: 12000` — PointTransformer тяжелее PointNet++ на крупных сценах, поэтому train-crop здесь заметно скромнее дефолта K3DXYZ.
- `feature_keys: pos,heights` — сохраняем тот же входной смысл, что и в `pointnet++` и `pointnext`, без RGB/интенсивности.

**Важно**
- Этот backbone использует CUDA pointops из [openpoints/cpp/pointops](/home/researcher/dev/PointNeXt/openpoints/cpp/pointops/setup.py#L1), так что для реального обучения нужен рабочий GPU и собранные расширения.
- Для `test()` в `examples/segmentation/main.py` нужен variable-path; я его тоже совместил с этим конфигом.

**Быстрый запуск**
```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/pointtransformer.yaml
```
