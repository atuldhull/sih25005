/// Centralized configuration for Demo Camera Mode.
///
/// ---------------------------------------------------------------------------
/// DEMO MODE ON   (emulator / prototype):
///   The capture screens render realistic bundled cow media instead of a
///   live camera feed. This is for demonstrating the capture workflow on
///   the Android/iOS emulator where there is usually no real camera.
///
/// DEMO MODE OFF  (physical device / real field worker):
///   Every capture screen uses the real CameraController / CameraPreview,
///   takePicture(), startVideoRecording() and stopVideoRecording() exactly
///   as before.
///
/// The mode is chosen at RUNTIME from Settings - see [CaptureSourceService].
/// ---------------------------------------------------------------------------
class DemoCameraConfig {
  /// True when the capture screens should use bundled photographs.
  ///
  /// This was `static const bool enabled = true`, which meant switching to the
  /// real camera was a recompile and a full restart - and that shipping the
  /// wrong constant to a judging table would have the app photograph a bundled
  /// cow while pointed at a real one.
  ///
  /// It is a plain getter over a mutable field on purpose: every existing
  /// `if (DemoCameraConfig.enabled)` in the four capture screens keeps working
  /// untouched, so making the mode switchable did not mean editing the capture
  /// flow itself. [CaptureSourceService] owns the value and persists it.
  static bool get enabled => _enabled;
  static bool _enabled = true;

  /// Set by [CaptureSourceService] only. Nothing else should write this.
  static set enabledInternal(bool value) => _enabled = value;

  /// Duration of the walking-video recording window.
  static const int videoDurationSeconds = 8;

  /// Asset keys resolved in priority order by [DemoMediaService].
  ///
  /// If a photorealistic asset is not present in the bundle the service
  /// falls back to the bundled placeholder image so the demo flow still
  /// works end-to-end.
  static const Map<DemoMediaKind, String> assetKeys = {
    DemoMediaKind.earTag: 'assets/cow_demo/cow_tag.jpg',
    DemoMediaKind.side: 'assets/cow_demo/cow_side.jpg',
    DemoMediaKind.rear: 'assets/cow_demo/cow_rear.jpg',
    DemoMediaKind.walkingVideo: 'assets/cow_demo/cow_walking.mp4',
  };

  /// Placeholder used when a photorealistic asset is missing.
  static const String fallbackImageAsset = 'assets/demo/cow_ear_tag.png';

  /// File name used when writing demo media into the app's temp dir.
  static String tempFileName(DemoMediaKind kind) {
    switch (kind) {
      case DemoMediaKind.earTag:
        return 'demo_ear_tag.jpg';
      case DemoMediaKind.side:
        return 'demo_side.jpg';
      case DemoMediaKind.rear:
        return 'demo_rear.jpg';
      case DemoMediaKind.walkingVideo:
        return 'demo_walking.mp4';
    }
  }
}

/// The kind of demo media used by each capture step.
enum DemoMediaKind { earTag, side, rear, walkingVideo }
