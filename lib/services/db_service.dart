import 'package:path/path.dart';
import 'package:sqflite/sqflite.dart';

import '../models/capture_session.dart';

class DbService {
  static Database? _database;

  static Future<Database> get database async {
    if (_database != null) return _database!;

    final dbPath = await getDatabasesPath();

    final path = join(dbPath, 'pashu_scorer.db');

    _database = await openDatabase(
      path,
      version: 2,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            animal_id TEXT,
            side_photo_path TEXT,
            rear_photo_path TEXT,
            video_path TEXT,
            result_json TEXT,
            status TEXT,
            captured_at TEXT
          )
        ''');

        await db.execute('''
          CREATE TABLE animals_cache (
            id TEXT PRIMARY KEY,
            name TEXT,
            profile_json TEXT
          )
        ''');
      },
      onUpgrade: (db, oldVersion, newVersion) async {
        if (oldVersion < 2) {
          await db.execute('''
            CREATE TABLE animals_cache (
              id TEXT PRIMARY KEY,
              name TEXT,
              profile_json TEXT
            )
          ''');
        }
      },
    );

    return _database!;
  }

  static Future<String> insertSession(CaptureSession session) async {
    final db = await database;

    final id = DateTime.now().millisecondsSinceEpoch.toString();

    await db.insert('sessions', {
      'id': id,
      'animal_id': session.tagId,
      'side_photo_path': session.sidePhotoPath,
      'rear_photo_path': session.rearPhotoPath,
      'video_path': session.videoPath,
      'result_json': null,
      'status': 'pending',
      'captured_at': DateTime.now().toIso8601String(),
    });

    return id;
  }

  static Future<List<Map<String, dynamic>>> getPendingSessions() async {
    final db = await database;

    return await db.query(
      'sessions',
      where: 'status = ?',
      whereArgs: ['pending'],
    );
  }

  static Future<void> markSynced(String id, String resultJson) async {
    final db = await database;

    await db.update(
      'sessions',
      {'status': 'synced', 'result_json': resultJson},
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  static Future<List<Map<String, dynamic>>> getAllSessions() async {
    final db = await database;

    return await db.query('sessions', orderBy: 'captured_at DESC');
  }

  /// Read-only helper (Step 4.3): returns all sessions for one animal.
  ///
  /// Used to correlate a History row with its locally stored SQLite session
  /// so the offline scorecard can be opened from `result_json`. This is a
  /// pure read — it never inserts, updates, or deletes sessions.
  static Future<List<Map<String, dynamic>>> getSessionsByAnimal(
    String animalId,
  ) async {
    final db = await database;

    return await db.query(
      'sessions',
      where: 'animal_id = ?',
      whereArgs: [animalId],
      orderBy: 'captured_at DESC',
    );
  }
}
