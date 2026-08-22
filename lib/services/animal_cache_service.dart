import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:sqflite/sqflite.dart';

import 'db_service.dart';

/// Caches animal profiles in SQLite for offline fallback.
///
/// The cache is purely additive — it never modifies or reads the
/// `sessions` table and never touches capture data.
class AnimalCacheService {
  /// Saves (inserts or updates) an animal profile in the cache.
  ///
  /// The animal [id] is used as the primary key, so calling this again
  /// with the same id replaces the previously cached profile.
  static Future<void> saveAnimal(
    String id,
    Map<String, dynamic> profile,
  ) async {
    try {
      final db = await DbService.database;

      await db.insert('animals_cache', {
        'id': id,
        'name': _extractName(profile),
        'profile_json': jsonEncode(profile),
      }, conflictAlgorithm: ConflictAlgorithm.replace);
    } catch (e) {
      debugPrint('AnimalCacheService.saveAnimal failed: $e');
    }
  }

  /// Reads a cached animal profile by [id].
  ///
  /// Returns `null` if no cached animal exists, or if the cached entry is
  /// malformed. Never throws.
  static Future<Map<String, dynamic>?> getAnimal(String id) async {
    try {
      final db = await DbService.database;

      final rows = await db.query(
        'animals_cache',
        where: 'id = ?',
        whereArgs: [id],
        limit: 1,
      );

      if (rows.isEmpty) return null;

      final profileJson = rows.first['profile_json'] as String?;
      if (profileJson == null || profileJson.isEmpty) return null;

      final decoded = jsonDecode(profileJson);
      if (decoded is Map<String, dynamic>) {
        return decoded;
      }
      return null;
    } catch (e) {
      debugPrint('AnimalCacheService.getAnimal failed: $e');
      return null;
    }
  }

  /// Extracts a display name for the cache row if the profile provides one.
  static String? _extractName(Map<String, dynamic> profile) {
    final name = profile['name'];
    if (name is String && name.isNotEmpty) return name;

    final breed = profile['breed'];
    if (breed is String && breed.isNotEmpty) return breed;

    return null;
  }
}
