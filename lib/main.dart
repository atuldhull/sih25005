import 'dart:async';

import 'package:flutter/material.dart';

import 'screens/capture/scan_tag_screen.dart';
import 'screens/records/records_screen.dart';
import 'services/sync_service.dart';
import 'widgets/sync_status_badge.dart';

void main() {
  runApp(const PashuScorerApp());
  // Start the global sync service without blocking app rendering.
  // Safe even if the backend is offline or SQLite is empty.
  unawaited(SyncService.instance.start());
}

class PashuScorerApp extends StatelessWidget {
  const PashuScorerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Pashu Scorer',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;

  late final List<Widget> _screens = [
    // Scan tab: sync status badge above the existing capture UI.
    const Column(
      children: [
        SyncStatusBadge(),
        Expanded(child: ScanTagScreen()),
      ],
    ),

    RecordsScreen(
      // "Start New Scoring Session" on an animal profile returns the user
      // to the Day 2 capture flow (Scan tab). The tag ID cannot be
      // pre-filled without modifying the protected ScanTagScreen, so the
      // user re-enters it during capture.
      onStartScoring: () {
        setState(() {
          _selectedIndex = 0;
        });
      },
    ),

    const Center(
      child: Text(
        'Settings\nComing soon',
        textAlign: TextAlign.center,
        style: TextStyle(fontSize: 20),
      ),
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _selectedIndex, children: _screens),

      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,

        onDestinationSelected: (index) {
          setState(() {
            _selectedIndex = index;
          });
        },

        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.qr_code_scanner),
            label: 'Scan',
          ),

          NavigationDestination(
            icon: Icon(Icons.folder_outlined),
            selectedIcon: Icon(Icons.folder),
            label: 'Records',
          ),

          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}
