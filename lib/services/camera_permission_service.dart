import 'package:flutter/foundation.dart';
import 'package:permission_handler/permission_handler.dart';

/// Asks for the permissions the capture screens need, at the moment they need
/// them.
///
/// WHY THIS EXISTS
/// The manifest declares CAMERA, and permission_handler was already a
/// dependency, but nothing in the app ever called it. On Android 6 and later a
/// declared permission is NOT a granted one - it has to be requested at
/// runtime - so CAMERA sat at `granted=false` and every real-camera capture
/// failed silently. Nothing threw; the preview simply never appeared.
///
/// It went unnoticed because DemoCameraConfig.enabled is true, and demo mode
/// uses bundled images instead of the camera. The moment that flag is flipped
/// for a physical device - which is the whole point of it - the app would have
/// stopped being able to take a photograph, on every phone.
///
/// Verified on the emulator: before this, `dumpsys package` reported
///     android.permission.CAMERA: granted=false
///     android.permission.RECORD_AUDIO: granted=false
/// and the only way to get a capture working was `adb shell pm grant`, which
/// is not something a farmer has.
class CameraPermissionService {
  /// Requests camera access, and microphone too when [forVideo] is set.
  ///
  /// Returns true when everything needed is granted. Never throws: a refusal
  /// is an answer, and the caller shows its own message rather than crashing
  /// in the middle of a capture.
  static Future<bool> ensure({bool forVideo = false}) async {
    try {
      final wanted = <Permission>[
        Permission.camera,
        if (forVideo) Permission.microphone,
      ];

      // Only ask for what is not already granted, so a farmer who has used the
      // app before is not prompted again on every screen.
      final missing = <Permission>[];
      for (final p in wanted) {
        if (!await p.isGranted) missing.add(p);
      }
      if (missing.isEmpty) return true;

      final results = await missing.request();
      return results.values.every((s) => s.isGranted);
    } catch (e) {
      // permission_handler throws on platforms that do not implement it
      // (desktop, web). Treat that as "nothing is blocking us" rather than
      // refusing to open a camera that needs no permission there.
      debugPrint('CameraPermissionService: $e');
      return true;
    }
  }
}
