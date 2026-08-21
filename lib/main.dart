import 'dart:async';

import 'package:flutter/material.dart';

import 'screens/alerts/alerts_screen.dart';
import 'screens/assistant/assistant_home_screen.dart';
import 'screens/capture/scan_tag_screen.dart';
import 'screens/records/records_screen.dart';
import 'screens/settings/settings_screen.dart';
import 'services/settings_service.dart';
import 'services/sync_service.dart';
import 'widgets/sync_status_badge.dart';

Future<void> main() async {
  // The saved server address has to be applied before anything makes a
  // request, or the first upload after a restart goes to the old host.
  WidgetsFlutterBinding.ensureInitialized();
  await SettingsService.restore();

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

    // Feature (i): ask about one animal, answered from her record alone,
    // in English, Hindi or Kannada.
    const AssistantHomeScreen(),

    // The veterinary officer's escalation feed. It lives in the app rather
    // than only on the server because the officer is the person who acts on
    // it, and they are in the field too.
    const AlertsScreen(),

    const SettingsScreen(),
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

        destinations: [
          const NavigationDestination(
            icon: Icon(Icons.qr_code_scanner),
            label: 'Scan',
          ),

          const NavigationDestination(
            icon: Icon(Icons.folder_outlined),
            selectedIcon: Icon(Icons.folder),
            label: 'Records',
          ),

          const NavigationDestination(
            icon: Icon(Icons.chat_bubble_outline),
            selectedIcon: Icon(Icons.chat_bubble),
            label: 'Assistant',
          ),

          const NavigationDestination(
            icon: Icon(Icons.notifications_none),
            selectedIcon: Icon(Icons.notifications),
            label: 'Alerts',
          ),

          NavigationDestination(
            // The count of captures still waiting rides on the Settings icon,
            // because that is where they can be sent by hand.
            icon: ValueListenableBuilder<int>(
              valueListenable: SyncService.instance.pending,
              builder: (context, count, child) => count == 0
                  ? child!
                  : Badge(label: Text('$count'), child: child),
              child: const Icon(Icons.settings_outlined),
            ),
            selectedIcon: const Icon(Icons.settings),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}
