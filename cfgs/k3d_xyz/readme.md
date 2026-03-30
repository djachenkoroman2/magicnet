# default.yaml

Файл [default.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/default.yaml#L4) задаёт базовые параметры датасета, загрузки данных, обучения и логирования для K3DXYZ. Смысл сверил по [k3d_xyz.py](/home/researcher/dev/PointNeXt/openpoints/dataset/k3d_xyz/k3d_xyz.py#L116), [data_util.py](/home/researcher/dev/PointNeXt/openpoints/dataset/data_util.py#L146), [main.py](/home/researcher/dev/PointNeXt/examples/segmentation/main.py#L248) и [pointnet++.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/pointnet++.yaml#L1). `null` здесь означает “не задано / отключено / без ограничения”.

**Датасет**
- `dataset.common.NAME`: имя класса датасета в registry; здесь создаётся `K3DXYZ`.
- `dataset.common.data_root`: корень данных; ожидаются папки `raw/`, `splits/`, `processed/` внутри `data/k3d_xyz`.
- `dataset.common.voxel_size`: размер вокселя для downsample перед обучением/валидацией; `0.2` уменьшает плотность облака точек.
- `dataset.common.label_values`: “сырые” метки из файлов датасета; внутри пайплайна они ремапятся из `[1..8]` в индексы `[0..7]`.
- `dataset.common.class_names`: человекочитаемые имена классов в том же порядке, что и `label_values`; используются в логах и метриках.
- `dataset.train.split`: брать список сцен из `splits/train.txt`.
- `dataset.train.voxel_max`: максимум точек в одном train-сэмпле после voxelize; если точек больше, берётся crop до `24000`.
- `dataset.train.loop`: виртуально повторять train-набор `30` раз за эпоху; длина датасета становится `число_сцен * 30`.
- `dataset.train.presample`: `False` значит train-сцены режутся на лету, поэтому в разных эпохах подвыборка может отличаться.
- `dataset.val.split`: брать сцены из `splits/val.txt`.
- `dataset.val.voxel_max`: `null` значит не ограничивать сцену по числу точек.
- `dataset.val.presample`: `True` значит один раз подготовить и закэшировать предобработанные val-сцены в `.pkl`, чтобы валидация была быстрее и стабильнее.
- `dataset.test.split`: брать сцены из `splits/test.txt`.
- `dataset.test.voxel_max`: `null` значит не резать тестовые сцены по лимиту точек.
- `dataset.test.presample`: `False` значит тестовые данные читать динамически, без статического `pkl`-кэша.
- `feature_keys`: какие поля склеивать во вход модели; `pos,heights` = `XYZ + height`, то есть 4 входных канала.
- `num_classes`: число классов сегментации; должно совпадать с длиной `label_values`.
- `batch_size`: batch size для train loader.
- `val_batch_size`: batch size для валидации; отдельно вынесен, потому что val обычно тяжелее по памяти.
- `dataloader.num_workers`: число процессов `DataLoader` для подготовки батчей.

**Трансформации**
- `datatransforms.train`: пайплайн аугментаций для train: `PointsToTensor` переводит массивы в `torch.Tensor`, `PointCloudScaling` случайно масштабирует, `PointCloudXYZAlign` центрирует XY и опускает минимум по Z к нулю, `PointCloudRotation` вращает, `PointCloudJitter` добавляет шум.
- `datatransforms.val`: пайплайн для val без случайных аугментаций, только перевод в тензоры и выравнивание.
- `datatransforms.kwargs.gravity_dim`: какая ось считается “высотой”; `2` означает `z`.
- `datatransforms.kwargs.scale`: диапазон случайного масштаба `[0.9, 1.1]`.
- `datatransforms.kwargs.angle`: пределы поворота в долях `pi` по осям `[x, y, z]`; `[0, 0, 1]` значит вращать только вокруг `z` в диапазоне `[-pi, pi]`.
- `datatransforms.kwargs.jitter_sigma`: стандартное отклонение шума для jitter.
- `datatransforms.kwargs.jitter_clip`: обрезка шума по модулю; шум не выйдет за `±0.02`.

**Обучение**
- `val_fn`: имя функции валидации. Но в текущем `examples/segmentation/main.py` для K3DXYZ фактически используется обычный `validate` автоматически, а не `cfg.val_fn`.
- `ignore_index`: индекс метки, которую надо игнорировать в метриках; `null` значит игнорируемых классов нет.
- `epochs`: количество эпох обучения.
- `cls_weighed_loss`: если `True`, loss получит веса классов по частотам train-набора, чтобы частично компенсировать дисбаланс.
- `criterion_args.NAME`: тип функции потерь; здесь `CrossEntropy`.
- `criterion_args.label_smoothing`: сглаживание меток; `0.2` уменьшает “жёсткость” target-ов и работает как регуляризация.
- `optimizer.NAME`: оптимизатор; здесь `AdamW`.
- `optimizer.weight_decay`: weight decay для оптимизатора.
- `sched`: тип scheduler-а learning rate; `cosine`.
- `warmup_epochs`: число эпох warmup; `0` значит warmup отключён.
- `min_lr`: минимальный LR, ниже которого cosine scheduler не опустится.
- `lr`: стартовый/base learning rate.
- `grad_norm_clip`: ограничение нормы градиента перед `optimizer.step`; помогает против взрывов градиента.
- `use_voting`: флаг дополнительной валидации “с голосованием” после обучения; в текущем коде практический эффект почти нулевой, потому что `validate()` не использует `num_votes`.

**IO**
- `log_dir`: имя для логов в конфиге, но в текущем коде оно потом перезаписывается реальным `run_dir`, так что почти не влияет.
- `save_freq`: частота сохранения промежуточных checkpoint-ов; `-1` значит не делать milestone-сейвы по эпохам, оставлять только `latest` и `best`.
- `val_freq`: как часто запускать валидацию; `1` значит каждую эпоху.
- `wandb.project`: имя проекта в Weights & Biases.

**Что важно знать**
- `feature_keys: pos,heights` здесь согласован с [pointnet++.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/pointnet++.yaml#L5), где `in_channels: 4`.
- `wandb.project` сам по себе не включает W&B; для этого ещё нужен `wandb.use_wandb: False/True` из [cfgs/default.yaml](/home/researcher/dev/PointNeXt/cfgs/default.yaml#L35).
- `log_dir` в рантайме перезаписывается в [logger.py](/home/researcher/dev/PointNeXt/openpoints/utils/logger.py#L129).
- `val_fn` и `use_voting` выглядят как частично устаревшие поля для текущего segmentation pipeline.

## PointNeXt для K3DXYZ

Для K3DXYZ теперь есть три конфига PointNeXt:

- [pointnext-s.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/pointnext-s.yaml#L1) — стартовый и самый сбалансированный вариант.
- [pointnext-b.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/pointnext-b.yaml#L1) — глубже `S`, использует `dataset.train.voxel_max: 32000`.
- [pointnext-l.yaml](/home/researcher/dev/PointNeXt/cfgs/k3d_xyz/pointnext-l.yaml#L1) — самый тяжёлый из трёх, с более осторожным `dataset.train.voxel_max: 24000`.

Все три используют тот же вход `feature_keys: pos,heights`, то есть `4` канала на точку: `xyz + height`.

Быстрые команды запуска:

```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/pointnext-s.yaml
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/pointnext-b.yaml
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/pointnext-l.yaml
```

Если хочется сначала самый безопасный по ресурсам вариант, начинай с `pointnext-s.yaml`, потом переходи к `pointnext-b.yaml`, и только после этого к `pointnext-l.yaml`.




# Логи

Для K3DXYZ логи у тебя лежат в `log/k3d_xyz/<run_name>/`.

Смотреть последний лог в реальном времени:
```bash
tail -f "$(ls -1t log/k3d_xyz/*/*.log | head -n 1)"
```

Показать последние 100 строк:
```bash
tail -n 100 "$(ls -1t log/k3d_xyz/*/*.log | head -n 1)"
```

Посмотреть, какие вообще есть логи:
```bash
find log/k3d_xyz -type f -name '*.log'
```

Сейчас у тебя есть, например, такой лог:
```bash
tail -f log/k3d_xyz/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD.log
```


# Tensorboard
Для `TensorBoard` по всем K3DXYZ-запускам:

```bash
tensorboard --logdir log/k3d_xyz --port 6006
```

Для конкретного текущего запуска:

```bash
tensorboard --logdir log/k3d_xyz/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD --port 6006
```

Потом открывай в браузере:
```bash
http://localhost:6006
```


# Чекпоинты

Чекпоинты посмотреть так:

```bash
ls -lh log/k3d_xyz/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD/checkpoint
```

У тебя сейчас есть:
- `..._ckpt_best.pth`
- `..._ckpt_latest.pth`

Если хочешь быстро посмотреть, какая эпоха записана в чекпоинте:

```bash
python -c "import torch; p='log/k3d_xyz/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD/checkpoint/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD_ckpt_best.pth'; ckpt=torch.load(p, map_location='cpu'); print('keys=', ckpt.keys()); print('epoch=', ckpt.get('epoch')); print('best_val=', ckpt.get('best_val'))"
```

И то же для `latest`:

```bash
python -c "import torch; p='log/k3d_xyz/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD/checkpoint/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD_ckpt_latest.pth'; ckpt=torch.load(p, map_location='cpu'); print('epoch=', ckpt.get('epoch')); print('best_val=', ckpt.get('best_val'))"
```


Вот готовые команды под твой текущий run.

Продолжить обучение с последнего чекпоинта:
```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/pointnet++.yaml \
  mode resume \
  pretrained_path log/k3d_xyz/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD/checkpoint/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD_ckpt_latest.pth
```

Провалидировать лучший чекпоинт на `val`:
```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/pointnet++.yaml \
  mode val \
  pretrained_path log/k3d_xyz/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD/checkpoint/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD_ckpt_best.pth
```

Запустить `test` с лучшим чекпоинтом по `test.txt`:
```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/pointnet++.yaml \
  mode test \
  pretrained_path log/k3d_xyz/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD/checkpoint/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD_ckpt_best.pth
```

Если хочешь прогнать `test()` не по `test.txt`, а по `val.txt`, добавь:
```bash
dataset.test.split val
```

То есть так:
```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyz/pointnet++.yaml \
  mode test \
  pretrained_path log/k3d_xyz/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD/checkpoint/k3d_xyz-train-pointnet++-ngpus1-20260327-213143-oYmXhrZLpqcZRVBipjvTyD_ckpt_best.pth \
  dataset.test.split val
```

Коротко:
- `ckpt_latest.pth` — для `resume`
- `ckpt_best.pth` — для `val` и обычно для `test`

