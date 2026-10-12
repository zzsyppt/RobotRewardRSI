"""PhysX contact history sampled at the configured sensor period.

Lab 3's SensorBase is lazy even for stateful sensors. History and mode durations
must consume every scheduled sample, independently of reward/observation reads.
The upstream force, timing and reset kernels remain the sole implementation.
"""

from isaaclab_physx.sensors.contact_sensor.contact_sensor import ContactSensor


class SampledContactSensor(ContactSensor):
    def update(self, dt: float, force_recompute: bool = False) -> None:
        super().update(
            dt,
            force_recompute=(
                force_recompute or self.cfg.history_length > 0 or self.cfg.track_air_time
            ),
        )
