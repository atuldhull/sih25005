import 'package:flutter/material.dart';

import 'screens/debug/mock_scorecard_debug_screen.dart';

/// Step 4.4 — Debug-only entry point for mock scorecard verification.
///
/// Run with:
///   flutter run -t lib/main_mock_debug.dart
///
/// This opens the mock scorecard verification route directly. It uses the
/// same visual theme as the real app but does NOT start the Day 3 sync
/// service and does NOT mount the normal Scan/Records home screen.
///
/// The normal entry point (lib/main.dart) and the normal Scan → capture →
/// save flow are completely untouched.
void main() {
  runApp(
    MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Pashu Scorer — Mock Debug',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const MockScorecardDebugScreen(),
    ),
  );
}
