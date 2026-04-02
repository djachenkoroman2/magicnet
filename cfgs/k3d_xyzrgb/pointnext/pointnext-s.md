# pointnext-s.yaml

Файл [pointnext-s.yaml](/home/researcher/dev/magicnet/cfgs/k3d_xyzrgb/pointnext/pointnext-s.yaml) наследует базовые настройки из [default.yaml](/home/researcher/dev/magicnet/cfgs/k3d_xyzrgb/default.yaml).

Это тот же `PointNeXt-S`, что и в `k3d_xyz`, но уже для цветного датасета `K3DXYZRGB`.

Главные отличия:

- `dataset.common.NAME: K3DXYZRGB`
- `feature_keys: pos,x,heights`
- `encoder_args.in_channels: 7`
- train/val transforms включают `ChromaticAutoContrast`, `ChromaticDropGPU`, `ChromaticNormalize`

Вход модели теперь состоит из:

- `xyz`
- `rgb`
- `height`

То есть всего `7` каналов на точку.

Остальная архитектура PointNeXt-S сохранена без изменений:

- `blocks: [1, 1, 1, 1, 1]`
- `sa_layers: 2`
- `sa_use_res: True`
- `width: 32`

Быстрый запуск:

```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyzrgb/pointnext/pointnext-s.yaml
```
