# dgcnn.yaml

Файл [dgcnn.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/dgcnn.yaml#L1) не является полностью автономным: он загружается вместе с [default.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/default.yaml#L4) через `cfg.load(..., recursive=True)`. То есть этот YAML переопределяет архитектуру, low-memory параметры для DGCNN, размеры батча и `dataset.train.voxel_max`; всё остальное берётся из `default.yaml`.

**Что задаёт этот файл**
- `model` — какую модель строить.
- `batch_size` — размер train batch.
- `val_batch_size` — размер validation batch.
- `step_per_update` — сколько mini-batch накапливать до шага оптимизатора.
- `dataset.train.voxel_max` — сколько точек максимум оставлять в одном train-сэмпле после voxelize/crop.

**Параметры `model`**
- `model.NAME: BaseSeg` — общая обёртка сегментации из [base_seg.py](/home/researcher/dev/PointNeXt/openpoints/models/segmentation/base_seg.py#L15). Она собирает `encoder`, затем `head`. `decoder_args` здесь нет, значит декодера не будет.
- `model.encoder_args` — параметры backbone.
- `model.cls_args` — параметры финальной сегментационной головы.

**Параметры `encoder_args`**
- `encoder_args.NAME: DGCNN` — использовать backbone DGCNN из [dgcnn.py](/home/researcher/dev/PointNeXt/openpoints/models/backbone/dgcnn.py#L13).
- `encoder_args.in_channels: 4` — число входных каналов на точку. Для K3DXYZ это согласовано с `feature_keys: pos,heights` из `default.yaml`: `xyz + height = 4`.
- `encoder_args.channels: 64` — базовая ширина первых графовых блоков. С этого числа начинается рост каналов внутри DGCNN.
- `encoder_args.embed_dim: 1024` — размер итогового pointwise embedding после fusion-блока. Так как `is_seg=True`, именно `1024` станет выходной размерностью энкодера.
- `encoder_args.n_blocks: 5` — число графовых блоков backbone. В текущей реализации это даёт 1 стартовый graph-conv и ещё `n_blocks - 2 = 3` динамических графовых блока.
- `encoder_args.conv: edge` — тип графовой свёртки. Здесь используется EdgeConv-стиль агрегации.
- `encoder_args.k: 20` — число соседей в kNN-графе для динамической графовой агрегации.
- `encoder_args.knn_max_cdist_mb: 128` — лимит на размер одного чанка для `torch.cdist` внутри kNN. Вместо полного `N x N` расстояния поиск соседей режется на куски, чтобы не ловить OOM на больших облаках.
- `encoder_args.is_seg: True` — режим сегментации. При нём `DGCNN.out_channels = embed_dim`, а не `embed_dim * 2`, как в классификации.
- `encoder_args.norm_args.norm: bn` — использовать batch normalization в блоках DGCNN.
- `encoder_args.act_args.act: leakyrelu` — активация LeakyReLU.
- `encoder_args.act_args.negative_slope: 0.2` — коэффициент отрицательной ветки LeakyReLU.
- `encoder_args.conv_args.order: conv-norm-act` — порядок операций в conv-блоках.

**Что это означает по форме сети**
- Вход: `4` канала на точку.
- Backbone строит динамический kNN-граф с `k=20`.
- После нескольких graph-conv блоков признаки со всех стадий конкатенируются.
- Затем fusion-блок сжимает их в `1024` каналов на точку.
- Поскольку это `BaseSeg`, эти `1024` канала автоматически передаются в `SegHead` как `in_channels` из [base_seg.py](/home/researcher/dev/PointNeXt/openpoints/models/segmentation/base_seg.py#L32).

**Параметры `cls_args`**
- `cls_args.NAME: SegHead` — финальная голова сегментации из [base_seg.py](/home/researcher/dev/PointNeXt/openpoints/models/segmentation/base_seg.py#L92).
- `cls_args.num_classes: 8` — сеть выдаёт 8 логитов на точку, по одному на класс K3DXYZ.
- `cls_args.mlps: [512, 256]` — структура головы будет такой: `1024 -> 512 -> 256 -> 8`.
- `cls_args.act_args.act: leakyrelu` — LeakyReLU внутри головы.
- `cls_args.act_args.negative_slope: 0.2` — коэффициент отрицательной ветки LeakyReLU в голове.

**Что не задано в `cls_args`, но важно**
- `in_channels` не указан специально: `BaseSeg` подставляет его автоматически из энкодера.
- `dropout` не указан, значит у `SegHead` останется значение по умолчанию `0.5`.
- `norm_args` не указан, значит у `SegHead` останется дефолтный `bn1d`.

**Параметры батча**
- `batch_size: 1` — train batch size. Значение специально занижено, потому что DGCNN для крупных облаков быстро упирается в VRAM.
- `val_batch_size: 1` — validation batch size. Маленький размер здесь разумен, потому что DGCNN тяжёлый по памяти.
- `step_per_update: 2` — градиентное накопление. Оптимизатор делает шаг раз в 2 итерации, так что эффективный batch остаётся ближе к `2`, даже если физически в GPU помещается только `1`.

**Параметры датасета**
- `dataset.train.voxel_max: 6000` — при обучении после voxelize/crop в один сэмпл попадёт максимум `6000` точек. Это низкопамятный дефолт для DGCNN, чтобы запуск не падал на полном `torch.cdist`.

**Что этот файл наследует из `default.yaml`**
- `dataset.common.NAME: K3DXYZ`
- `dataset.common.data_root`, `voxel_size`, `label_values`, `class_names`
- `feature_keys: pos,heights`
- `num_classes: 8`
- `datatransforms`
- `epochs`, `optimizer`, `sched`, `lr`
- `log_dir`, `val_freq`, `wandb.project`
- train/val/test split-настройки, кроме переопределённого `dataset.train.voxel_max`

**Практический смысл**
- Это DGCNN-конфиг для point cloud segmentation на K3DXYZ.
- Он тяжелее `pointnet++.yaml`, но лучше использует локальную графовую структуру.
- За это платишь меньшим `voxel_max`, меньшим `batch_size` и более медленным kNN из-за chunked `cdist`.

## Шпаргалка по DGCNN для K3DXYZ

Файл [dgcnn.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/dgcnn.yaml#L1) задаёт DGCNN-конфигурацию для сегментации K3DXYZ. В отличие от PointNet++, DGCNN строит динамический граф соседей, поэтому он обычно тяжелее по VRAM и заметно медленнее на больших облаках.

### Что крутить для качества

- `dataset.train.voxel_max` вверх: `6000 -> 8000/12000`. Это самый прямой способ дать DGCNN больше точек на сцену.
- `dataset.common.voxel_size` вниз: `0.2 -> 0.15/0.1`. Больше геометрических деталей, но сильнее растут память и время.
- `model.encoder_args.k` вверх: `20 -> 24/32`. Каждый узел графа видит больше соседей, что может улучшать локальную геометрию.
- `model.encoder_args.channels` вверх: `64 -> 96`. Модель становится шире и мощнее.
- `model.encoder_args.embed_dim` вверх: `1024 -> 1536`. Богаче итоговое point-wise представление.
- `model.encoder_args.n_blocks` вверх: `5 -> 6`. Сеть становится глубже, но растёт цена обучения.
- `model.cls_args.mlps` шире: `[512, 256] -> [512, 256, 128]` или `[768, 256]`.
- `epochs` вверх: `100 -> 150/200`.
- `cls_weighed_loss: True`, если классы в K3DXYZ несбалансированы.
- `criterion_args.label_smoothing` можно снизить до `0.1`, если видишь, что модель недоучивается.

### Что крутить для скорости и памяти

- `dataset.train.voxel_max` вниз: `6000 -> 4000/3000`. Это самый эффективный способ разгрузить DGCNN.
- `batch_size` здесь уже минимальный: `1`. Если OOM сохраняется, сначала опускай `voxel_max` и `k`, а не пытайся уменьшить batch дальше.
- `step_per_update` вверх: `2 -> 4`, если хочешь сохранить более внятный effective batch при очень жёстких ограничениях по памяти.
- `dataset.common.voxel_size` вверх: `0.2 -> 0.25/0.3`. Меньше точек после voxelize, быстрее граф и меньше память.
- `model.encoder_args.k` вниз: `20 -> 16`. Соседей меньше, граф строится дешевле.
- `model.encoder_args.channels` вниз: `64 -> 32`. Существенно облегчает backbone.
- `model.encoder_args.embed_dim` вниз: `1024 -> 512`. Уменьшает размер финального признакового пространства.
- `model.encoder_args.n_blocks` вниз: `5 -> 4`. Укорачивает backbone.
- `val_freq` реже: `1 -> 5`, если хочешь меньше пауз на валидацию.

### Что менять в первую очередь

- Для качества без больших архитектурных переделок: `dataset.train.voxel_max`, `dataset.common.voxel_size`, `model.encoder_args.k`.
- Для экономии VRAM: `batch_size`, `dataset.train.voxel_max`, `dataset.common.voxel_size`.
- Для ускорения без сильной потери качества: сначала `voxel_max` и `val_freq`, а уже потом `channels` и `embed_dim`.

### Практичные стартовые наборы для DGCNN

- Быстрый старт:
  - `dataset.train.voxel_max: 4000`
  - `batch_size: 1`
  - `step_per_update: 2`
  - `model.encoder_args.k: 16`
  - `epochs: 80`

- Сбалансированный режим:
  - `dataset.train.voxel_max: 8000`
  - `batch_size: 1`
  - `step_per_update: 2`
  - `model.encoder_args.k: 20`
  - `epochs: 120`

- Качество:
  - `dataset.common.voxel_size: 0.15`
  - `dataset.train.voxel_max: 12000`
  - `batch_size: 1`
  - `step_per_update: 2`
  - `model.encoder_args.k: 24`
  - `model.encoder_args.embed_dim: 1536`
  - `epochs: 160`

### Как читать поведение DGCNN

- Если `train_miou` растёт, а `val_miou` почти стоит: скорее всего, не хватает регуляризации или модель переобучается.
- Если и `train`, и `val` растут медленно: сначала попробуй увеличить `voxel_max` и `epochs`, а не сразу делать сеть шире.
- Если обучение очень тяжёлое по памяти: DGCNN почти всегда лучше сначала разгружать по числу точек, а не по числу эпох.
- Если модель стала слишком медленной: первым делом уменьши `k` или `voxel_max`.


## Готовые пресеты для DGCNN

Ниже 3 готовых пресета для запуска [dgcnn.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/dgcnn.yaml#L1) на K3DXYZ через CLI-overrides. Они сделаны в том же стиле, что и пресеты для PointNet++, но с поправкой на то, что DGCNN заметно тяжелее по динамическому графу.

Пресеты ниже тоже остаются low-memory. Если у тебя GPU свободна почти полностью, их можно делать агрессивнее уже после первого стабильного запуска.

Базовая форма запуска:

```bash
CUDA_VISIBLE_DEVICES=0 uv run examples/segmentation/main.py --cfg cfgs/k3d_xyz/dgcnn.yaml ...
```

### `fast`

Подходит для быстрой проверки пайплайна, smoke-test обучения и дешёвых итераций по гиперпараметрам.

Что меняется:
- меньше точек в train-сэмпле
- более грубая voxel-сетка
- меньше соседей в графе
- реже валидация
- меньше эпох

```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/dgcnn.yaml \
  dataset.common.voxel_size 0.25 \
  dataset.train.voxel_max 4000 \
  batch_size 1 \
  step_per_update 2 \
  epochs 80 \
  val_freq 5 \
  criterion_args.label_smoothing 0.1 \
  model.encoder_args.k 16
```

Ожидаемый эффект:
- самый дешёвый по времени режим
- проще переживает ограниченную VRAM
- качество обычно ниже, чем у `balanced` и `quality`

### `balanced`

Подходит как основной стартовый режим для DGCNN, если нужен разумный компромисс между качеством и вычислительной ценой.

Что меняется:
- чуть больше точек на сцену
- умеренный `k`
- немного длиннее обучение
- мягче label smoothing

```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/dgcnn.yaml \
  dataset.train.voxel_max 8000 \
  batch_size 1 \
  step_per_update 2 \
  epochs 120 \
  criterion_args.label_smoothing 0.1 \
  model.encoder_args.k 20
```

Ожидаемый эффект:
- самый практичный первый кандидат для DGCNN
- обычно заметно стабильнее `fast`
- всё ещё терпим по памяти по сравнению с более агрессивными настройками

### `quality`

Подходит, когда DGCNN уже стабильно запускается и ты хочешь выжать из него максимум на K3DXYZ.

Что меняется:
- более плотная voxel-сетка
- больше точек в train-crop
- больше соседей в графе
- крупнее embedding
- длиннее обучение
- включены веса классов
- ниже learning rate и добавлен warmup

```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/dgcnn.yaml \
  dataset.common.voxel_size 0.15 \
  dataset.train.voxel_max 12000 \
  batch_size 1 \
  step_per_update 2 \
  epochs 160 \
  lr 0.005 \
  warmup_epochs 5 \
  cls_weighed_loss True \
  criterion_args.label_smoothing 0.1 \
  model.encoder_args.k 24 \
  model.encoder_args.embed_dim 1536
```

Ожидаемый эффект:
- лучший шанс получить сильные метрики на текущем backbone
- самый тяжёлый режим по памяти и времени
- если появляется OOM, сначала снижай `batch_size`, потом `dataset.train.voxel_max`

### Как выбрать быстро

- Если сейчас важно просто запустить обучение и проверить, что всё работает: используй `fast`.
- Если нужен нормальный рабочий baseline для DGCNN: используй `balanced`.
- Если уже есть запас по ресурсам и нужна максимальная точность: используй `quality`.

### На что смотреть после первого прогона

- Если `train_miou` и `val_miou` обе низкие: сначала попробуй поднять `epochs` и `dataset.train.voxel_max`.
- Если `train_miou` уходит вверх, а `val_miou` почти не растёт: посмотри на `label_smoothing`, `cls_weighed_loss` и общий размер модели.
- Если DGCNN слишком медленный: сначала уменьшай `model.encoder_args.k`.
- Если не хватает памяти: сначала уменьшай `batch_size`, затем `dataset.train.voxel_max`, потом повышай `dataset.common.voxel_size`.
