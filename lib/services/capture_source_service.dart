import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';

import 'db_service.dart';
import 'demo_camera_config.dart';

/// Where a capture's photographs come from.
enum CaptureSource {
  /// Bundled photographs. Runs anywhere, including an emulator with no
  /// camera, and is what the app has always done.
  demo,

  /// The device's real camera.
  camera,
}

/// Chooses the capture source at runtime, and persists the choice.
///
/// The mode used to be a compile-time constant, so showing the real camera
/// meant a rebuild. That is no use at a judging table, where the question
/// "does it work on an actual animal?" arrives without warning.
///
/// Picking an existing photograph from the gallery is deliberately NOT a mode:
/// it is available in both, as a button beside the shutter. A judge handing
/// over their phone with a photo of a cow on it should not require anyone to
/// change a setting first.
class CaptureSourceService {
  static const String _key = 'capture_source';

  /// The current source. Settings listens to this; the capture screens read
  /// [DemoCameraConfig.enabled], which this keeps in step.
  static final ValueNotifier<CaptureSource> source =
      ValueNotifier<CaptureSource>(CaptureSource.demo);

  /// Loads the saved choice. Never throws - a corrupt row leaves the app on
  /// demo, which is the mode that works everywhere.
  static Future<void> restore() async {
    try {
      final saved = await DbService.getSetting(_key);
      if (saved == 'camera') {
        _apply(CaptureSource.camera);
      } else {
        _apply(CaptureSource.demo);
      }
    } catch (e) {
      debugPrint('CaptureSourceService.restore failed: $e');
      _apply(CaptureSource.demo);
    }
  }

  static Future<void> set(CaptureSource value) async {
    _apply(value);
    try {
      await DbService.setSetting(
        _key,
        value == CaptureSource.camera ? 'camera' : 'demo',
      );
    } catch (e) {
      debugPrint('CaptureSourceService.set failed to persist: $e');
    }
  }

  static void _apply(CaptureSource value) {
    source.value = value;
    DemoCameraConfig.enabledInternal = value == CaptureSource.demo;
  }
}

/// Picks an existing photograph or video off the device.
///
/// Available on every capture screen in every mode, because it is the one
/// route that works on an emulator with no camera, on a phone whose camera
/// permission was declined, and when someone wants to score an animal from a
/// photograph they already had.
///
/// Every returned file is COPIED into the app's own cache directory first. A
/// gallery URI is not a stable file path - it can be a content:// handle the
/// app may not be able to re-read later, and the upload queue may not get to
/// it for hours.
class MediaPicker {
  static final ImagePicker _picker = ImagePicker();

  /// Returns a path to a copied still image, or null if the user backed out.
  static Future<String?> pickImage() async {
    return _pick(() => _picker.pickImage(source: ImageSource.gallery),
        'picked_image');
  }

  /// Returns a path to a copied video, or null if the user backed out.
  static Future<String?> pickVideo() async {
    return _pick(() => _picker.pickVideo(source: ImageSource.gallery),
        'picked_video');
  }

  static Future<String?> _pick(
    Future<XFile?> Function() choose,
    String prefix,
  ) async {
    try {
      final picked = await choose();
      if (picked == null) return null;      // backed out - not an error

      final dir = await getTemporaryDirectory();
      final ext = _extensionOf(picked.name, picked.path);
      final stamp = DateTime.now().millisecondsSinceEpoch;
      final target = File('${dir.path}/${prefix}_$stamp$ext');
      await target.writeAsBytes(await picked.readAsBytes(), flush: true);
      return target.path;
    } catch (e) {
      debugPrint('MediaPicker failed: $e');
      return null;
    }
  }

  static String _extensionOf(String name, String path) {
    for (final candidate in [name, path]) {
      final dot = candidate.lastIndexOf('.');
      if (dot > 0 && dot < candidate.length - 1) {
        final ext = candidate.substring(dot).toLowerCase();
        // Guard against a query string riding along on a content:// path.
        if (ext.length <= 5 && !ext.contains('/')) return ext;
      }
    }
    return '.jpg';
  }
}
