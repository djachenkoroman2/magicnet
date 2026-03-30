# pointnet++.yaml

Ниже разбор [pointnet++.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/pointnet++.yaml#L1) по фактическому поведению в [PointNet2Encoder/Decoder](/home/researcher/dev/PointNeXt/openpoints/models/backbone/pointnetv2.py#L149) и [BaseSeg/SegHead](/home/researcher/dev/PointNeXt/openpoints/models/segmentation/base_seg.py#L15).

**Модель**
- `model.NAME: BaseSeg` — общая обёртка сегментации: `encoder -> decoder -> head`.
- `encoder_args.NAME: PointNet2Encoder` — энкодер PointNet++ для извлечения иерархических признаков.
- `encoder_args.in_channels: 4` — число входных каналов на точку. Здесь это согласовано с `feature_keys: pos,heights` из [default.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/default.yaml#L25): `x,y,z,height`.
- `encoder_args.width: null` — базовая ширина. В этом конфиге фактически не используется, потому что каналы заданы явно через `mlps`.
- `encoder_args.strides: [4, 4, 4, 4]` — на каждой из 4 стадий число точек уменьшается примерно в 4 раза через sampling.
- `encoder_args.layers: 3` — число слоёв внутри блока. Здесь практически тоже не влияет, потому что структура уже зафиксирована в `mlps`.
- `encoder_args.use_res: False` — без residual-связей внутри локальных SA-блоков.
- `encoder_args.mlps` — каналы MLP по стадиям:
  - 1 стадия: `4 -> 32 -> 32 -> 64`
  - 2 стадия: `64 -> 64 -> 64 -> 128`
  - 3 стадия: `128 -> 128 -> 128 -> 256`
  - 4 стадия: `256 -> 256 -> 256 -> 512`
- `encoder_args.radius: 0.4` — базовый радиус соседей для `ballquery`. Так как задан скаляром, код развернёт его по стадиям в `0.4, 0.8, 1.6, 3.2`.
- `encoder_args.num_samples: 32` — максимум соседей в локальной области. Разворачивается в `32, 32, 32, 32`.
- `encoder_args.sampler: fps` — downsampling через farthest point sampling.
- `encoder_args.aggr_args.NAME: convpool` — тип локальной агрегации: сначала свёртки по соседям, потом pooling.
- `encoder_args.aggr_args.feature_type: dp_fj` — в локальные признаки подаются относительные координаты соседей `dp` и их признаки `fj`.
- `encoder_args.aggr_args.anisotropic: False` — в текущей реализации `PointNet2 + convpool` напрямую не используется.
- `encoder_args.aggr_args.reduction: max` — агрегация по соседям через max-pooling.
- `encoder_args.group_args.NAME: ballquery` — соседи выбираются по радиусу.
- `encoder_args.group_args.use_xyz: True` — в текущем коде это поле напрямую не читается; геометрия и так попадает в блок через `feature_type: dp_fj`.
- `encoder_args.conv_args.order: conv-norm-act` — порядок операций в conv-блоках.
- `encoder_args.act_args.act: relu` — функция активации ReLU.
- `encoder_args.norm_args.norm: bn` — batch normalization в энкодере.
- `decoder_args.NAME: PointNet2Decoder` — декодер Feature Propagation, который поднимает coarse-признаки обратно к исходным точкам.
- `decoder_args.fp_mlps` — каналы FP-модулей декодера, от грубых уровней к более детальным: `[[128,128,128], [256,128], [256,256], [256,256]]`.
- `cls_args.NAME: SegHead` — финальная голова сегментации.
- `cls_args.num_classes: 8` — 8 логитов на точку, по одному на класс.
- `cls_args.in_channels: null` — это нормально: [BaseSeg](/home/researcher/dev/PointNeXt/openpoints/models/segmentation/base_seg.py#L32) сам подставит сюда выход декодера, то есть `128`.

**Трансформации**
- `datatransforms.train` — train-аугментации: перевод в тензоры, масштабирование, выравнивание облака, случайный поворот, jitter.
- `datatransforms.val` — только перевод в тензоры и выравнивание, без случайных искажений.
- `datatransforms.kwargs.gravity_dim: 2` — ось `z` считается высотой.
- `datatransforms.kwargs.scale: [0.9, 1.1]` — случайный масштаб в диапазоне 0.9–1.1.
- `datatransforms.kwargs.angle: [0, 0, 1]` — случайный поворот только вокруг `z`, в диапазоне `[-pi, pi]`.
- `datatransforms.kwargs.jitter_sigma: 0.005` — сила гауссова шума.
- `datatransforms.kwargs.jitter_clip: 0.02` — обрезка шума по модулю.

**Итог**
- По сути это классический PointNet++ для сегментации: `4` входных канала, энкодер с ширинами `64/128/256/512`, потом FP-декодер и сегментационная голова `128 -> 128 -> 8`.
- В этом конкретном YAML реально важные поля: `in_channels`, `strides`, `mlps`, `radius`, `num_samples`, `sampler`, `aggr_args`, `fp_mlps`, `num_classes`.
- `width`, `layers`, `aggr_args.anisotropic` и `group_args.use_xyz` здесь либо не влияют, либо почти не влияют в текущей реализации.


## Шпаргалка по быстродействию

**Шпаргалка**
Если цель качество, сначала трогай [default.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/default.yaml#L4) и [pointnet++.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/pointnet++.yaml#L1) в таком порядке:

- `dataset.common.voxel_size` вниз: `0.2 -> 0.1/0.05`. Это обычно самый прямой способ сохранить больше геометрии, но резко растит память и время.
- `dataset.train.voxel_max` вверх: `24000 -> 32000/48000`. Модель видит больше точек из сцены, часто это помогает.
- `encoder_args.num_samples` вверх: `32 -> 48/64`. Каждый центр видит больше соседей, локальные признаки богаче.
- `encoder_args.radius` под плотность сцены. Если сцены разреженные, можно увеличить `0.4 -> 0.6`; если плотные и “слипаются” классы, иногда лучше уменьшить.
- `encoder_args.mlps` шире: например `64/128/256/512 -> 96/192/384/768`. Это уже дорогой, но сильный рычаг.
- `epochs` вверх: `100 -> 150/200`, если модель ещё не сошлась.
- `cls_weighed_loss: True`, если классы несбалансированы.
- `criterion_args.label_smoothing` аккуратно: `0.2` может быть многовато. Если видишь недообучение, попробуй `0.1` или `0.05`.
- Аугментации: если модель переобучается, чуть усилить `scale` и `jitter`; если недоучивается на геометрии, наоборот ослабить.

Если цель память и скорость, самые эффективные ручки такие:

- `dataset.common.voxel_size` вверх: `0.2 -> 0.3/0.4`.
- `dataset.train.voxel_max` вниз: `24000 -> 16000/12000`.
- `batch_size` вниз, если упираешься в VRAM.
- `val_batch_size` держать маленьким, как сейчас `1`.
- `encoder_args.num_samples` вниз: `32 -> 16/24`.
- `encoder_args.mlps` уже: например `32/64/128/256` вместо текущих.
- `epochs` вниз, если нужен быстрый прогон.
- `val_freq` реже: `1 -> 5`, чтобы меньше тратить времени на валидацию.
- `dataset.val.presample: True` оставлять включённым; это хороший компромисс для ускорения валидации.
- `dataset.train.loop` вниз, если хочешь короче эпоху по wall-clock времени.

**Что менять в первую очередь**
- Для качества без сильного роста модели: `voxel_size`, `voxel_max`, `num_samples`.
- Для экономии VRAM без ломки архитектуры: `voxel_size`, `voxel_max`, `batch_size`.
- Для максимального качества, если есть ресурсы: потом уже `mlps` и `epochs`.

**Практичные стартовые наборы**
- “Чуть лучше качество”: `voxel_size: 0.1`, `voxel_max: 32000`, `num_samples: 48`, `epochs: 150`.
- “Легче и быстрее”: `voxel_size: 0.3`, `voxel_max: 16000`, `num_samples: 16`, `batch_size: 4`.
- “Если сильный дисбаланс классов”: `cls_weighed_loss: True`, `label_smoothing: 0.1`.

Если хочешь, я могу следующим сообщением дать тебе уже готовые 3 пресета конфигов: `fast`, `balanced`, `quality`.


## Готовые пресеты для PointNet++

Ниже 3 практичных пресета для запуска [pointnet++.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/pointnet++.yaml#L1) на K3DXYZ без создания новых YAML-файлов. Они опираются на то, что `main.py` умеет переопределять вложенные поля через CLI-аргументы.

Базовая форма запуска:

```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/pointnet++.yaml ...
```

Эти пресеты я специально сделал "безопасными": они меняют только самые полезные и предсказуемые ручки, не ломая базовую архитектурную совместимость конфига.

## `fast`

Подходит для быстрых итераций, smoke-test обучения и подбора пайплайна, когда важнее скорость, чем максимум качества.

Что меняется:
- меньше точек на сцену
- более грубая voxel-сетка
- меньше соседей в локальной агрегации
- реже валидация
- меньше эпох

```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/pointnet++.yaml \
  dataset.common.voxel_size 0.3 \
  dataset.train.voxel_max 16000 \
  batch_size 8 \
  epochs 80 \
  val_freq 5 \
  criterion_args.label_smoothing 0.1 \
  encoder_args.num_samples 16
```

Ожидаемый эффект:
- самый быстрый из трёх режимов
- заметно дешевле по памяти
- качество обычно ниже `balanced` и `quality`

## `balanced`

Подходит как основной рабочий стартовый режим, если нужен хороший компромисс между временем обучения, памятью и качеством.

Что меняется:
- чуть больше контекста по точкам
- мягче label smoothing
- немного дольше обучение
- остальные параметры остаются близки к базовым

```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/pointnet++.yaml \
  dataset.common.voxel_size 0.15 \
  dataset.train.voxel_max 32000 \
  batch_size 6 \
  epochs 140 \
  criterion_args.label_smoothing 0.1 \
  encoder_args.num_samples 32 \
  encoder_args.radius 0.45
```

Ожидаемый эффект:
- обычно лучший первый кандидат для реальной тренировки
- ощутимо лучше сохраняет геометрию сцены, чем `fast`
- требует больше VRAM и времени, чем базовый конфиг

## `quality`

Подходит, когда важен максимум качества и есть запас по VRAM и времени обучения.

Что меняется:
- более плотная voxel-сетка
- больше точек в train-crop
- больше соседей
- чуть больше радиус локального поиска
- более длинное обучение
- включены веса классов
- ниже learning rate и добавлен warmup

```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/pointnet++.yaml \
  dataset.common.voxel_size 0.1 \
  dataset.train.voxel_max 48000 \
  batch_size 2 \
  epochs 180 \
  lr 0.005 \
  warmup_epochs 5 \
  cls_weighed_loss True \
  criterion_args.label_smoothing 0.1 \
  encoder_args.radius 0.5 \
  encoder_args.num_samples 48
```

Ожидаемый эффект:
- лучший шанс выжать максимум из текущей архитектуры
- самый дорогой по времени и памяти режим
- если упираешься в VRAM, первым делом уменьшай `batch_size`, потом `dataset.train.voxel_max`

## Как выбрать быстро

- Если сейчас главное запустить цикл и проверить, что всё учится: используй `fast`.
- Если нужен нормальный боевой старт без экстремальных затрат: используй `balanced`.
- Если уже всё работает и хочешь добивать метрики: используй `quality`.

## На что смотреть после первого прогона

- Если модель недоучивается: увеличивай `epochs`, `dataset.train.voxel_max`, `encoder_args.num_samples`.
- Если модель переобучается: ослабляй модель не первым делом; сначала посмотри на `label_smoothing`, аугментации и `cls_weighed_loss`.
- Если не хватает памяти: сначала уменьшай `batch_size`, потом `dataset.train.voxel_max`, потом повышай `dataset.common.voxel_size`.
- Если обучение слишком медленное: первым делом переходи на `fast` или подними `val_freq`.
