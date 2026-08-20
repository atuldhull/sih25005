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

  /// Stops the service and releases the connectivity subscription.
  Future<void> dispose() async {
    await _connectivitySubscription?.cancel();
    _connectivitySubscription = null;
    _isStarted = false;
    _apiService.close();
  }
}
