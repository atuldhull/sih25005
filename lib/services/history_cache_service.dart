import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:sqflite/sqflite.dart';

import 'db_service.dart';

/// Caches animal scoring history in SQLite for offline fallback.
///
/// The cache is purely additive — it creates its own `history_cache` table
/// lazily (CREATE TABLE IF NOT EXISTS) on the shared DbService database.
/// It never modifies the `sessions` table, never touches capture data,
/// and does not require a database migration.
class HistoryCacheService {
  /// Saves (replaces) the cached history for an animal.
  ///
  /// The animal [id] is used as the primary key, so calling this again
  /// with the same id replaces the previously cached history.
  static Future<void> saveHistory(
    String id,
    List<Map<String, dynamic>> sessions,
  ) async {
    try {
      final db = await DbService.database;
      await _ensureTable(db);

      await db.insert('history_cache', {
        'id': id,
        'history_json': jsonEncode(sessions),
        'updated_at': DateTime.now().toIso8601String(),
      }, conflictAlgorithm: ConflictAlgorithm.replace);
    } catch (e) {
      debugPrint('HistoryCacheService.saveHistory failed: $e');
    }
  }

  /// Reads the cached history for an animal.
  ///
  /// Returns an empty list if no cached history exists, or if the cached
  /// entry is malformed. Never throws.
  static Future<List<Map<String, dynamic>>> getHistory(String id) async {
    try {
      final db = await DbService.database;
      await _ensureTable(db);

      final rows = await db.query(
        'history_cache',
        where: 'id = ?',
        whereArgs: [id],
        limit: 1,
      );

      if (rows.isEmpty) return [];

      final historyJson = rows.first['history_json'] as String?;
      if (historyJson == null || historyJson.isEmpty) return [];

      final decoded = jsonDecode(historyJson);
      if (decoded is List) {
        return decoded.whereType<Map<String, dynamic>>().toList();
      }
      return [];
    } catch (e) {
      debugPrint('HistoryCacheService.getHistory failed: $e');
      return [];
    }
  }

  /// Creates the `history_cache` table if it does not already exist.
  ///
  /// This is purely additive and requires no migration — the table is
  /// created lazily on first use.
  static Future<void> _ensureTable(Database db) async {
    await db.execute('''
      CREATE TABLE IF NOT EXISTS history_cache (
        id TEXT PRIMARY KEY,
        history_json TEXT,
        updated_at TEXT
      )
    ''');
  }
}
