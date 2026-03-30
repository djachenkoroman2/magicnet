# pointnext-s.yaml

Файл [pointnext-s.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/pointnext-s.yaml#L1) не автономный: он загружается вместе с [default.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/default.yaml#L4) через `cfg.load(..., recursive=True)`. То есть этот YAML переопределяет архитектуру `PointNeXt-S` и локально фиксирует train/val аугментации, а параметры датасета, оптимизатора, scheduler-а и логирования приходят из `default.yaml`.

**Что задаёт этот файл**
- `model` — полная архитектура сегментационной сети.
- `datatransforms` — train/val аугментации для K3DXYZ.

**Параметры `model`**
- `model.NAME: BaseSeg` — общая обёртка сегментации из [base_seg.py](/home/researcher/dev/PointNeXt/openpoints/models/segmentation/base_seg.py#L15): `encoder -> decoder -> head`.
- `encoder_args.NAME: PointNextEncoder` — backbone PointNeXt из [pointnext.py](/home/researcher/dev/PointNeXt/openpoints/models/backbone/pointnext.py#L311).
- `encoder_args.blocks: [1, 1, 1, 1, 1]` — число блоков по стадиям. Это конфигурация `PointNeXt-S`: по одному блоку на стадию.
- `encoder_args.strides: [1, 4, 4, 4, 4]` — первая стадия работает без downsample, дальше число точек уменьшается примерно в 4 раза на каждой стадии.
- `encoder_args.sa_layers: 2` — в set-abstraction блоке используется более глубокий локальный MLP. Это одна из характерных черт `PointNeXt-S`.
- `encoder_args.sa_use_res: True` — residual-связь внутри SA-блока включена; для `S`-варианта это ожидаемое поведение.
- `encoder_args.width: 32` — базовая ширина сети. По мере downsample ширина автоматически растёт внутри энкодера.
- `encoder_args.in_channels: 4` — число входных каналов на точку. Здесь это `xyz + height`, что согласовано с `feature_keys: pos,heights` из [default.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/default.yaml#L25).
- `encoder_args.expansion: 4` — коэффициент расширения во внутренних inverted residual MLP-блоках.
- `encoder_args.radius: 0.4` — базовый радиус `ballquery`. Так как задан скаляром, код развернёт его по стадиям с увеличением масштаба: `0.4, 0.8, 1.6, 3.2, 6.4`.
- `encoder_args.nsample: 32` — максимум соседей на локальную область. Это значение разворачивается на все стадии.
- `encoder_args.aggr_args.feature_type: dp_fj` — агрегация использует относительную геометрию соседей и их признаки.
- `encoder_args.aggr_args.reduction: max` — локальная агрегация по соседям идёт через max-pooling.
- `encoder_args.group_args.NAME: ballquery` — соседей выбираем по радиусу, а не через kNN.
- `encoder_args.group_args.normalize_dp: True` — относительные координаты соседей нормализуются внутри grouper-а.
- `encoder_args.conv_args.order: conv-norm-act` — порядок операций в conv-блоках.
- `encoder_args.act_args.act: relu` — активация ReLU.
- `encoder_args.norm_args.norm: bn` — batch normalization в backbone.
- `decoder_args.NAME: PointNextDecoder` — декодер Feature Propagation для возврата coarse-признаков к исходным точкам.
- `cls_args.NAME: SegHead` — голова сегментации.
- `cls_args.num_classes: 8` — по одному логиту на каждый класс K3DXYZ.
- `cls_args.in_channels: null` — это нормально: [BaseSeg](/home/researcher/dev/PointNeXt/openpoints/models/segmentation/base_seg.py#L32) сам подставит размер выхода декодера.
- `cls_args.norm_args.norm: bn` — batch normalization в голове.

**Что это означает по форме сети**
- Вход: `4` канала на точку (`x, y, z, height`).
- Энкодер PointNeXt строит 5 стадий с локальной агрегацией по радиусу.
- Затем `PointNextDecoder` поднимает признаки обратно к плотности исходных точек.
- После этого `SegHead` выдаёт `8` логитов на каждую точку сцены.

**Параметры `datatransforms`**
- `train: [PointsToTensor, PointCloudScaling, PointCloudXYZAlign, PointCloudRotation, PointCloudJitter]` — базовый train-pipeline для K3DXYZ без RGB-аугментаций, потому что в датасете только геометрия и метка.
- `val: [PointsToTensor, PointCloudXYZAlign]` — валидация без случайных искажений.
- `gravity_dim: 2` — ось `z` считается высотой.
- `scale: [0.9, 1.1]` — случайный масштаб.
- `angle: [0, 0, 1]` — поворот только вокруг `z`.
- `jitter_sigma: 0.005` — сила гауссова шума.
- `jitter_clip: 0.02` — ограничение шума по модулю.

**Чем `PointNeXt-S` отличается от PointNet++ на K3DXYZ**
- Вместо `PointNet2Encoder` используется [PointNextEncoder](/home/researcher/dev/PointNeXt/openpoints/models/backbone/pointnext.py#L311) с inverted residual MLP-блоками.
- Первая стадия работает без downsample, поэтому сеть дольше сохраняет плотные локальные детали.
- У `PointNeXt-S` включены `sa_layers: 2` и `sa_use_res: True`, чего нет в базовом PointNet++.
- Декодер остаётся компактным: [PointNextDecoder](/home/researcher/dev/PointNeXt/openpoints/models/backbone/pointnext.py#L460) сам использует `encoder_channel_list` и не требует вручную выписывать `fp_mlps`.

**Практический смысл**
- Это хороший стартовый `PointNeXt`-конфиг для K3DXYZ, если хочется модель современнее PointNet++, но без резкого роста вычислительной цены.
- По вычислительной цене он обычно ближе к “сбалансированному” режиму: заметно тяжелее `pointnet++`, но существенно легче, чем `PointNeXt-L`.
- Если по памяти всё стабильно, дальше логично смотреть на [pointnext-b.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/pointnext-b.yaml#L1) и [pointnext-l.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/pointnext-l.yaml#L1).

**Быстрый запуск**
```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/pointnext-s.yaml
```

**Валидация чекпоинта**
```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/pointnext-s.yaml \
  mode val \
  pretrained_path /path/to/pointnext_s_ckpt_best.pth
```
