# pointnet++.yaml

Это прямой аналог `cfgs/k3d_xyz/pointnet++/pointnet++.yaml`, но для `K3DXYZRGB`.

Главные отличия от версии без цвета:

- используется `dataset.common.NAME: K3DXYZRGB`
- базовый `default.yaml` подаёт `feature_keys: pos,x,heights`
- `encoder_args.in_channels: 7`
- в train/val pipeline добавлены RGB-аугментации и нормализация цвета

Здесь `7` каналов означают:

- `xyz`
- `rgb`
- `height`

Архитектура PointNet++ и все размеры декодера оставлены теми же, что в `k3d_xyz`, чтобы сравнение между датасетами было честным.

Быстрый запуск:

```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyzrgb/pointnet++/pointnet++.yaml
```
