import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';

import '../../models/scoring_result.dart';
import '../explain/scorecard_screen.dart';

/// Step 4.4 — Debug-only mock scorecard verification.
///
/// Loads `assets/mock/scoring_result.json` directly, parses it through the
/// real [ScoringResult.fromJson], and renders it with the real
/// [ScorecardScreen] so the complete explainability UI (20 traits →
/// category grouping → trait detail navigation) can be verified against a
/// fixed fixture.
///
/// This route is verification-only. It is not part of the normal Scan →
/// capture → save flow, never writes mock data into Records/History, and
/// never replaces real scoring data.
///
/// Note: the mock asset is deliberately NOT registered in pubspec.yaml
/// (protected file), so `rootBundle.loadString` cannot be used. This route
/// reads the file with `dart:io File` instead, which resolves relative to
/// the process working directory (the project root when running
/// `flutter run` from this repository).
class MockScorecardDebugScreen extends StatefulWidget {
  const MockScorecardDebugScreen({super.key});

  @override
  State<MockScorecardDebugScreen> createState() =>
      _MockScorecardDebugScreenState();
}

class _MockScorecardDebugScreenState extends State<MockScorecardDebugScreen> {
  /// Path of the mock fixture relative to the process working directory.
  static const String _mockAssetPath = 'assets/mock/scoring_result.json';

  late Future<ScoringResult> _loadFuture;

  @override
  void initState() {
    super.initState();
    _loadFuture = _loadMockScore();
  }

  /// Reads the mock JSON file, decodes it, and converts it to a
  /// [ScoringResult] through the real model factory. Every failure is
  /// converted into a user-friendly [_MockLoadException] — raw exception
  /// text is never surfaced to the UI.
  Future<ScoringResult> _loadMockScore() async {
    try {
      final file = File(_mockAssetPath);
      if (!await file.exists()) {
        throw const _MockLoadException('The mock fixture could not be found.');
      }

      final raw = await file.readAsString();
      final decoded = jsonDecode(raw);
      if (decoded is! Map<String, dynamic>) {
        throw const _MockLoadException(
          'The mock fixture is not a JSON object.',
        );
      }

      return ScoringResult.fromJson(decoded);
    } on _MockLoadException {
      rethrow;
    } on FormatException {
      throw const _MockLoadException(
        'The mock fixture contains malformed JSON.',
      );
    } on FileSystemException {
      throw const _MockLoadException(
        'The mock fixture file could not be read.',
      );
    } catch (_) {
      // Covers any ScoringResult.fromJson / type-cast failure.
      throw const _MockLoadException(
        'The mock fixture could not be parsed into a scoring result.',
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<ScoringResult>(
      future: _loadFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          );
        }

        final result = snapshot.data;
        if (result != null) {
          // Render the real explainability scorecard with the real model.
          return ScorecardScreen(result: result);
        }

        final error = snapshot.error;
        return Scaffold(
          appBar: AppBar(title: const Text('Mock Scorecard (Debug)')),
          body: Center(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: _MockLoadErrorCard(
                message: error is _MockLoadException
                    ? error.message
                    : 'The mock fixture could not be loaded.',
              ),
            ),
          ),
        );
      },
    );
  }
}

/// User-facing error for the debug loader. The message is neutral and never
/// contains raw exception details.
class _MockLoadException implements Exception {
  final String message;

  const _MockLoadException(this.message);
}

/// Neutral error card shown when the mock fixture cannot be loaded — no red
/// error state, no raw exception text.
class _MockLoadErrorCard extends StatelessWidget {
  final String message;

  const _MockLoadErrorCard({required this.message});

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: Colors.grey.shade300),
      ),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.info_outline, size: 40, color: Colors.grey.shade400),
            const SizedBox(height: 12),
            const Text(
              'Mock scorecard unavailable',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),
            Text(
              message,
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 13,
                color: Colors.grey.shade600,
                height: 1.4,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
