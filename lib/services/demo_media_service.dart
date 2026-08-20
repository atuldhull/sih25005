import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import 'demo_camera_config.dart';

/// Result of capturing a piece of demo media.
class DemoMediaResult {
  const DemoMediaResult({
    required this.assetKey,
    required this.filePath,
    this.preferredAssetMissing = false,
  });

  /// The asset key that was actually used (preferred or fallback).
  final String assetKey;

  /// Absolute path to the copied media file in the app temp directory.
  final String filePath;

  /// True when the preferred photorealistic asset is not bundled yet and a
  /// fallback (placeholder image) was used instead.
  final bool preferredAssetMissing;
}

/// Resolves and copies bundled cow demo media into the app temp directory.
///
/// The capture screens always hand a real file path to the quality gate and
/// to [CaptureSession.videoPath], so the existing
/// `Image.file(...)` / SQLite flow keeps working unchanged.
class DemoMediaService {
  static final Map<String, bool> _assetCache = {};

  // ============================================================
  // ASSET EXISTENCE
  // ============================================================

  /// Returns true when [assetKey] can be loaded from the bundle.
  ///
  /// Results are cached so screens can safely pick between the
  /// photorealistic asset and the bundled placeholder for `Image.asset`.
  static Future<bool> assetExists(String assetKey) async {
    final cached = _assetCache[assetKey];
    if (cached != null) {
      return cached;
    }

    var exists = true;
    try {
      await rootBundle.load(assetKey);
    } catch (_) {
      exists = false;
    }

    _assetCache[assetKey] = exists;
    return exists;
  }

  /// Resolves the best `Image.asset` key for a live camera preview.
  ///
  /// Prefers the photorealistic `assets/cow_demo/<kind>.jpg` asset and
  /// falls back to `assets/demo/cow_ear_tag.png` when it is not bundled.
  static Future<String> resolveImageAsset(DemoMediaKind kind) async {
    final preferred = DemoCameraConfig.assetKeys[kind]!;

    if (await assetExists(preferred)) {
      return preferred;
    }

    return DemoCameraConfig.fallbackImageAsset;
  }

  // ============================================================
  // CAPTURE DEMO MEDIA (copies bytes -> temp file)
  // ============================================================

  /// Copies the demo image for [kind] into the app temp directory and
  /// returns the resulting file path.
  ///
  /// This keeps `PhotoQualityScreen` (which uses `Image.file`) and the
  /// existing retake cleanup flow working without modification.
  static Future<DemoMediaResult> captureImage(DemoMediaKind kind) async {
    final dir = await getTemporaryDirectory();
    final file = File(
      '${dir.path}${Platform.pathSeparator}'
      '${DemoCameraConfig.tempFileName(kind)}',
    );

    final preferred = DemoCameraConfig.assetKeys[kind]!;
    final loaded = await _loadFirstAvailable([
      preferred,
      DemoCameraConfig.fallbackImageAsset,
    ]);

    await file.writeAsBytes(loaded.bytes, flush: true);

    return DemoMediaResult(
      assetKey: preferred,
      filePath: file.path,
      preferredAssetMissing: loaded.isFallback,
    );
  }

  /// Copies the demo walking video into the app temp directory.
  ///
  /// When `assets/cow_demo/cow_walking.mp4` is not bundled yet, the service
  /// writes the placeholder image bytes into the same temp file name so the
  /// session always receives a valid, existing file path for SQLite. The
  /// video screen inspects [DemoMediaResult.preferredAssetMissing] and shows
  /// the static placeholder instead of trying to play it.
  static Future<DemoMediaResult> captureVideo() async {
    final dir = await getTemporaryDirectory();
    final file = File(
      '${dir.path}${Platform.pathSeparator}'
      '${DemoCameraConfig.tempFileName(DemoMediaKind.walkingVideo)}',
    );

    final preferred = DemoCameraConfig.assetKeys[DemoMediaKind.walkingVideo]!;

    if (await assetExists(preferred)) {
      final bytes = await _loadAssetBytes(preferred);
      await file.writeAsBytes(bytes, flush: true);

      return DemoMediaResult(
        assetKey: preferred,
        filePath: file.path,
        preferredAssetMissing: false,
      );
    }

    // Placeholder fallback so the session path is always a real file.
    final fallbackBytes = await _loadAssetBytes(
      DemoCameraConfig.fallbackImageAsset,
    );
    await file.writeAsBytes(fallbackBytes, flush: true);

    return DemoMediaResult(
      assetKey: DemoCameraConfig.fallbackImageAsset,
      filePath: file.path,
      preferredAssetMissing: true,
    );
  }

  // ============================================================
  // INTERNAL HELPERS
  // ============================================================

  static Future<Uint8List> _loadAssetBytes(String assetKey) async {
    final data = await rootBundle.load(assetKey);
    return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  }

  static Future<({Uint8List bytes, bool isFallback})> _loadFirstAvailable(
    List<String> keys,
  ) async {
    for (var i = 0; i < keys.length; i++) {
      try {
        return (bytes: await _loadAssetBytes(keys[i]), isFallback: i != 0);
      } catch (_) {
        // Try the next candidate.
      }
    }

    // The fallback entry is always the last key and is declared in
    // pubspec.yaml, so it must be loadable.
    final bytes = await _loadAssetBytes(keys.last);
    return (bytes: bytes, isFallback: true);
  }
}
