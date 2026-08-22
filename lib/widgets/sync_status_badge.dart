import 'dart:async';

import 'package:flutter/material.dart';

import '../services/db_service.dart';

/// A compact, persistent indicator showing the number of capture sessions
/// waiting to sync to the backend.
///
/// The badge is a passive viewer of SQLite pending state — it does NOT
/// perform synchronization itself. [SyncService] owns all upload logic.
class SyncStatusBadge extends StatefulWidget {
  const SyncStatusBadge({super.key});

  @override
  State<SyncStatusBadge> createState() => _SyncStatusBadgeState();
}

class _SyncStatusBadgeState extends State<SyncStatusBadge> {
  Timer? _timer;
  int? _pendingCount;
  bool _hasError = false;

  @override
  void initState() {
    super.initState();
    _refresh();
    _timer = Timer.periodic(const Duration(seconds: 2), (_) => _refresh());
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _refresh() async {
    try {
      final pending = await DbService.getPendingSessions();
      if (!mounted) return;
      setState(() {
        _pendingCount = pending.length;
        _hasError = false;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _hasError = true;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final Widget content;

    if (_hasError) {
      content = _buildStatus(
        icon: Icons.cloud_off_outlined,
        label: 'Sync status unavailable',
        color: Colors.grey.shade600,
        background: Colors.grey.withValues(alpha: 0.10),
      );
    } else if (_pendingCount == null) {
      content = _buildStatus(
        icon: Icons.sync,
        label: 'Checking sync status…',
        color: Colors.grey.shade600,
        background: Colors.grey.withValues(alpha: 0.10),
      );
    } else if (_pendingCount == 0) {
      content = _buildStatus(
        icon: Icons.check_circle_outline,
        label: '✓ All synced',
        color: Colors.green.shade700,
        background: Colors.green.withValues(alpha: 0.12),
      );
    } else if (_pendingCount == 1) {
      content = _buildStatus(
        icon: Icons.cloud_upload_outlined,
        label: '1 session waiting to sync',
        color: Colors.blue.shade700,
        background: Colors.blue.withValues(alpha: 0.10),
      );
    } else {
      content = _buildStatus(
        icon: Icons.cloud_upload_outlined,
        label: '$_pendingCount sessions waiting to sync',
        color: Colors.blue.shade700,
        background: Colors.blue.withValues(alpha: 0.10),
      );
    }

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
      child: Center(child: content),
    );
  }

  Widget _buildStatus({
    required IconData icon,
    required String label,
    required Color color,
    required Color background,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: background,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 18, color: color),
          const SizedBox(width: 8),
          Text(
            label,
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w500,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}
