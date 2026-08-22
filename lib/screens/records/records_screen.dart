import 'package:flutter/material.dart';

import 'animal_profile_screen.dart';
import 'history_screen.dart';

/// Minimal Records screen (Day 3 placeholder replacement).
///
/// Lets the user enter an animal/tag ID and open the animal's profile
/// or scoring history. This keeps the History tab reachable without
/// altering the Scan tab capture flow.
class RecordsScreen extends StatefulWidget {
  /// Optional callback invoked when the user taps
  /// "Start New Scoring Session" on an animal profile.
  ///
  /// When null, the button on [AnimalProfileScreen] renders disabled.
  final VoidCallback? onStartScoring;

  const RecordsScreen({super.key, this.onStartScoring});

  @override
  State<RecordsScreen> createState() => _RecordsScreenState();
}

class _RecordsScreenState extends State<RecordsScreen> {
  final TextEditingController _controller = TextEditingController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _openHistory(String animalId) {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => HistoryScreen(animalId: animalId),
      ),
    );
  }

  void _openProfile(String animalId) {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => AnimalProfileScreen(
          animalId: animalId,
          onStartScoring: widget.onStartScoring,
        ),
      ),
    );
  }

  String? _validatedId() {
    final id = _controller.text.trim();
    if (id.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Enter an animal/tag ID first.')),
      );
      return null;
    }
    return id;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Animal Records')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text(
            'Enter the animal/tag ID to view its records.',
            style: TextStyle(fontSize: 14),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _controller,
            keyboardType: TextInputType.number,
            decoration: const InputDecoration(
              labelText: 'Animal / Tag ID',
              hintText: 'e.g. 356279812345',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: () {
                    final id = _validatedId();
                    if (id != null) _openHistory(id);
                  },
                  icon: const Icon(Icons.show_chart),
                  label: const Text('View History'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () {
                    final id = _validatedId();
                    if (id != null) _openProfile(id);
                  },
                  icon: const Icon(Icons.person_outline),
                  label: const Text('View Profile'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),
          Card(
            elevation: 0,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
              side: BorderSide(color: Colors.grey.shade300),
            ),
            child: const Padding(
              padding: EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Notes', style: TextStyle(fontWeight: FontWeight.w600)),
                  SizedBox(height: 8),
                  Text(
                    'History shows past scoring sessions and the weight '
                    'trend. Offline, the last cached history for an animal '
                    'is shown. Tap a past session to open its scorecard.',
                    style: TextStyle(fontSize: 13),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
