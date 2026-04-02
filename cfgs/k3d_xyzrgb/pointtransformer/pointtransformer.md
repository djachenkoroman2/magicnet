# pointtransformer.yaml

Это аналог `cfgs/k3d_xyz/pointtransformer/pointtransformer.yaml`, но для нового датасета `K3DXYZRGB`.

Что изменено относительно версии без RGB:

- `model.in_channels: 7`
- `feature_keys: pos,x,heights`
- `dataset.common.NAME: K3DXYZRGB`
- добавлены цветовые transforms и `ChromaticNormalize`

Смысл входа:

- `xyz`
- `rgb`
- `height`

Тот же `PointTransformer` по-прежнему работает в variable-length режиме, поэтому здесь сохраняются:

- `dataset.common.variable: True`
- `dataloader.collate_fn: concat_collate_fn`
- `batch_size: 1`
- `step_per_update: 2`

Быстрый запуск:

```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyzrgb/pointtransformer/pointtransformer.yaml
```
