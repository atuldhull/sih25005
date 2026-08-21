import 'dart:async';
import 'dart:convert';

import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:flutter/foundation.dart';

import '../models/capture_session.dart';
import 'api_service.dart';
import 'db_service.dart';

/// Orchestrates offline synchronization of pending capture sessions.
///
/// SyncService is responsible for:
/// - Detecting when network connectivity becomes available
/// - Reading pending sessions from SQLite
/// - Uploading them using [ApiService]
/// - Marking successful uploads as synced
/// - Leaving failed sessions pending
///
/// SyncService NEVER deletes a pending session.
class SyncService {
  /// Singleton instance.
  static final SyncService instance = SyncService._internal();

  SyncService._internal();

  ApiService _apiService = ApiService();

  StreamSubscription<List<ConnectivityResult>>? _connectivitySubscription;

  bool _isStarted = false;
  bool _isSyncing = false;

  /// How many captures are still waiting to reach the server.
  ///
  /// A ValueNotifier rather than a future because three separate places want
  /// to show it - the badge above the capture flow, Settings, and the saved
  /// screen - and polling SQLite from each of them would be wasteful.
  final ValueNotifier<int> pending = ValueNotifier<int>(0);

  /// Refreshes [pending] from SQLite. Never throws.
  Future<void> refreshPending() async {
    try {
      pending.value = await DbService.countPendingSessions();
    } catch (e) {
      debugPrint('SyncService.refreshPending failed: $e');
    }
  }

  /// Starts the service: subscribes to connectivity changes and
  /// immediately performs one initial sync attempt.
  ///
  /// Calling [start] multiple times is safe — only one connectivity
  /// subscription is ever created.
  Future<void> start() async {
    if (_isStarted) return;
    _isStarted = true;

    _apiService = ApiService();

    _connectivitySubscription = Connectivity().onConnectivityChanged.listen(
      _onConnectivityChanged,
    );

    await refreshPending();

    // Initial sync: handles the case where the app starts with
    // internet already available and pending sessions in SQLite.
    await syncPendingSessions();
  }

  /// Handles connectivity change events.
  ///
  /// If connectivity is NONE, does nothing.
  /// Otherwise, triggers a pending-session synchronization.
  void _onConnectivityChanged(List<ConnectivityResult> results) {
    if (results.isEmpty || results.contains(ConnectivityResult.none)) {
      return;
    }
    unawaited(syncPendingSessions());
  }

  /// Uploads all pending sessions and marks successful ones as synced.
  ///
  /// Failed uploads are left pending and will be retried on a future
  /// connectivity event or app startup.
  Future<void> syncPendingSessions() async {
    if (_isSyncing) return;
    _isSyncing = true;

    try {
      final pendingSessions = await DbService.getPendingSessions();

      for (final row in pendingSessions) {
        try {
          await _syncOne(row);
        } catch (e) {
          debugPrint('SyncService: failed to sync a session: $e');
        }
      }
    } catch (e) {
      debugPrint('SyncService.syncPendingSessions failed: $e');
    } finally {
      _isSyncing = false;
      await refreshPending();
    }
  }

  /// Uploads a single pending session row.
  ///
  /// On success, marks the session synced with the server result.
  /// On failure, leaves the session pending — it is never deleted
  /// and its status is never changed.
  Future<void> _syncOne(Map<String, dynamic> row) async {
    final id = row['id'] as String?;
    if (id == null || id.isEmpty) {
      debugPrint('SyncService: pending session row missing id');
      return;
    }

    final session = CaptureSession(
      tagId: row['animal_id'] as String?,
      sidePhotoPath: row['side_photo_path'] as String?,
      rearPhotoPath: row['rear_photo_path'] as String?,
      videoPath: row['video_path'] as String?,
      // Without this the queued retry uploads without the ear-tag close-up,
      // so a session that went offline scores strictly worse than the same
      // session uploaded immediately - every centimetre trait refused for
      // want of a scale.
      tagPhotoPath: row['tag_photo_path'] as String?,
    );

    final result = await _apiService.uploadSession(session);

    if (result == null) {
      // Upload failed — leave the session pending for a future retry.
      return;
    }

    try {
      await DbService.markSynced(id, jsonEncode(result));
    } catch (e) {
      debugPrint('SyncService: failed to mark session $id synced: $e');
    }
  }

  /// Uploads ONE session immediately and returns the server's scorecard.
  ///
  /// This is the path a capture takes the moment it is finished, while the
  /// farmer is still holding the phone in front of the animal. The queue is
  /// for retries; making the first attempt go through it would mean waiting
  /// on a connectivity event to see a result that was available at once.
  ///
  /// Returns null when the upload fails or the row has already synced with
  /// nothing stored. The session is left pending either way - never deleted,
  /// and never marked synced on a failure.
  Future<Map<String, dynamic>?> uploadSessionNow(String localId) async {
    // Refresh at both ends, not just on success.
    //
    // This used to call refreshPending() only after markSynced, so a capture
    // that FAILED to upload never moved the counter - and a failed upload is
    // precisely when something is waiting to be sent. Reproduced on the
    // emulator: with the server unreachable, the saved screen correctly said
    // "Saved on this phone. Not sent yet.", the database correctly held one
    // pending row, and Settings said "Nothing waiting" with Send now greyed
    // out. The one control that could have flushed the queue by hand was the
    // one the bug disabled.
    //
    // The leading refresh also makes the count include the row that was just
    // inserted, so the badge appears the moment a capture is saved rather
    // than after the upload attempt finishes.
    await refreshPending();
    try {
      final row = await DbService.getSessionById(localId);
      if (row == null) return null;

      // Already uploaded - hand back what was stored rather than posting the
      // same photographs a second time.
      final stored = row['result_json'] as String?;
      if (row['status'] == 'synced' && stored != null && stored.isNotEmpty) {
        final decoded = jsonDecode(stored);
        return decoded is Map<String, dynamic> ? decoded : null;
      }

      final result = await _apiService.uploadSession(
        CaptureSession(
          tagId: row['animal_id'] as String?,
          sidePhotoPath: row['side_photo_path'] as String?,
          rearPhotoPath: row['rear_photo_path'] as String?,
          videoPath: row['video_path'] as String?,
          tagPhotoPath: row['tag_photo_path'] as String?,
        ),
      );
      if (result == null) return null;

      await DbService.markSynced(localId, jsonEncode(result));
      return result;
    } catch (e) {
      debugPrint('SyncService.uploadSessionNow failed: $e');
      return null;
    } finally {
      await refreshPending();
    }
  }

  /// Stops the service and releases the connectivity subscription.
  Future<void> dispose() async {
    await _connectivitySubscription?.cancel();
    _connectivitySubscription = null;
    _isStarted = false;
    _apiService.close();
  }
}
