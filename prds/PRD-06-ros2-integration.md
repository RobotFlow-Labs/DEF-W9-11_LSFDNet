# PRD-06: ROS2 Integration

> Module: LSFDNet | Priority: P1
> Depends on: PRD-05
> Status: ⬜ Not started

## Objective
Wrap LSFDNet inference as a ROS2 node for ANIMA graph integration (topic I/O, launch files, and manifest mapping).

## Context (from paper)
LSFDNet targets maritime perception and naturally fits robotics/edge sensing pipelines where SWIR/LWIR arrive as synchronized streams.

**Paper reference**: Section 1 and Section 5 (practical maritime deployment motivation).

## Acceptance Criteria
- [ ] ROS2 node subscribes to SWIR + LWIR image topics and publishes fused output.
- [ ] Supports synchronized callbacks and timestamp propagation.
- [ ] Includes launch file and parameterized model path/device settings.
- [ ] Provides ANIMA-facing topic contract documentation.
- [ ] Smoke test runs in ROS2 simulation mode.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `ros2/anima_lsfdnet_node.py` | ROS2 node for pairwise fusion inference | deployment | ~220 |
| `ros2/launch/lsfdnet.launch.py` | Launch entrypoint | deployment | ~80 |
| `anima_module.yaml` | ANIMA module interface metadata | integration | ~80 |

## Architecture Detail (from paper)

### Inputs
```text
/swir/image_raw: sensor_msgs/Image
/lwir/image_raw: sensor_msgs/Image
```

### Outputs
```text
/lsfdnet/fused/image: sensor_msgs/Image
/lsfdnet/meta: std_msgs/String
```

### Algorithm
```python
class LSFDNetNode(Node):
    def on_synced_pair(self, swir_msg, lwir_msg):
        swir, lwir = ros_to_tensor(swir_msg, lwir_msg)
        fused, _ = self.model(swir, lwir)
        self.pub_fused.publish(tensor_to_ros(fused))
```

## Dependencies
```toml
rclpy = "ROS2"
cv_bridge = "ROS2"
message_filters = "ROS2"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| Runtime checkpoint | N/A | ROS2 param `model_path` | Baidu link |

## Test Plan
```bash
# ROS2 environment required
ros2 launch lsfdnet lsfdnet.launch.py
ros2 topic echo /lsfdnet/meta
```

## References
- Paper: Section 1, Section 5
- Depends on: PRD-05
- Feeds into: PRD-07
