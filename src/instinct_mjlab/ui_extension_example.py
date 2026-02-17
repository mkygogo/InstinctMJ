from __future__ import annotations

try:
  import omni.ext as _omni_ext
  import omni.ui as _omni_ui
except ModuleNotFoundError:
  _omni_ext = None
  _omni_ui = None


def some_public_function(x: int) -> int:
  print("[instinctlab] some_public_function was called with x: ", x)
  return x**x


if _omni_ext is not None and _omni_ui is not None:

  class ExampleExtension(_omni_ext.IExt):
    def on_startup(self, ext_id: str) -> None:
      print("[instinctlab] startup")

      self._count = 0
      self._window = _omni_ui.Window("My Window", width=300, height=300)
      with self._window.frame:
        with _omni_ui.VStack():
          label = _omni_ui.Label("")

          def on_click() -> None:
            self._count += 1
            label.text = f"count: {self._count}"

          def on_reset() -> None:
            self._count = 0
            label.text = "empty"

          on_reset()

          with _omni_ui.HStack():
            _omni_ui.Button("Add", clicked_fn=on_click)
            _omni_ui.Button("Reset", clicked_fn=on_reset)

    def on_shutdown(self) -> None:
      print("[instinctlab] shutdown")

else:

  class ExampleExtension:
    def on_startup(self, ext_id: str) -> None:
      raise RuntimeError("omni is required to run ExampleExtension.")

    def on_shutdown(self) -> None:
      return None


__all__ = ["ExampleExtension", "some_public_function"]
