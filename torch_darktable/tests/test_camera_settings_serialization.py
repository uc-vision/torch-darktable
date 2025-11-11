"""Test CameraSettings serialization/deserialization."""

from torch_darktable.pipeline.camera_settings import load_camera_settings_from_dir


def test_camera_settings_roundtrip():
  """Test that all camera settings can be serialized and deserialized."""
  camera_settings = load_camera_settings_from_dir()
  for _, settings in camera_settings.items():
    json_str = settings.model_dump_json()
    settings_copy = settings.__class__.model_validate_json(json_str)
    assert settings == settings_copy
