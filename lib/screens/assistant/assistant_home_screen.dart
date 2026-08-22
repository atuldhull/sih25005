import 'package:flutter/material.dart';

import '../../services/api_service.dart';
import 'chat_screen.dart';

/// Pick an animal, then ask about her.
///
/// The assistant answers strictly from one animal's record, so a chat has to
/// begin by choosing which animal. The list comes from the BPA records on the
/// server rather than from anything captured on this phone — a farmer can ask
/// about an animal that has never been scored here.
class AssistantHomeScreen extends StatefulWidget {
  const AssistantHomeScreen({super.key});

  @override
  State<AssistantHomeScreen> createState() => _AssistantHomeScreenState();
}

class _AssistantHomeScreenState extends State<AssistantHomeScreen> {
  final ApiService _api = ApiService();
  final TextEditingController _filter = TextEditingController();

  List<Map<String, dynamic>> _animals = const [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _filter.dispose();
    _api.close();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    final reply = await _api.getAnimals();
    if (!mounted) return;

    final error = reply['error'];
    setState(() {
      _loading = false;
      if (error != null) {
        _error = error.toString();
        return;
      }
      _animals = (reply['animals'] as List? ?? [])
          .whereType<Map>()
          .map((a) => Map<String, dynamic>.from(a))
          .toList();
    });
  }

  List<Map<String, dynamic>> get _visible {
    final q = _filter.text.trim().toLowerCase();
    if (q.isEmpty) return _animals;
    return _animals.where((a) {
      return [
        a['animal_id'],
        a['breed'],
        a['village'],
        a['species'],
      ].any((v) => v.toString().toLowerCase().contains(q));
    }).toList();
  }

  void _openChat(Map<String, dynamic> animal) {
    final id = animal['animal_id']?.toString();
    if (id == null || id.isEmpty) return;
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => ChatScreen(
          animalId: id,
          animalLabel: animal['breed']?.toString(),
        ),
      ),
    );
  }

  Future<void> _openTypedId() async {
    final id = _filter.text.trim();
    if (id.isEmpty) return;
    Navigator.of(context).push(
      MaterialPageRoute<void>(builder: (_) => ChatScreen(animalId: id)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Assistant'),
        actions: [
          IconButton(
            tooltip: 'Reload',
            onPressed: _loading ? null : _load,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
            child: TextField(
              controller: _filter,
              onChanged: (_) => setState(() {}),
              textInputAction: TextInputAction.search,
              onSubmitted: (_) => _openTypedId(),
              decoration: InputDecoration(
                hintText: 'Tag ID, breed or village',
                prefixIcon: const Icon(Icons.search),
                border: const OutlineInputBorder(),
                isDense: true,
                suffixIcon: _filter.text.isEmpty
                    ? null
                    : IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () => setState(_filter.clear),
                      ),
              ),
            ),
          ),
          Expanded(child: _body()),
        ],
      ),
    );
  }

  Widget _body() {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_error != null) {
      return _Retry(
        message: _error!,
        // A typed tag ID still works with the list unavailable: the chat
        // endpoint looks the animal up itself, so being offline for /animals
        // does not have to block the conversation.
        onRetry: _load,
        secondary: _filter.text.trim().isEmpty
            ? null
            : TextButton(
                onPressed: _openTypedId,
                child: Text('Ask about ${_filter.text.trim()} anyway'),
              ),
      );
    }

    final rows = _visible;
    if (rows.isEmpty) {
      return _Retry(
        message: 'No animal matches “${_filter.text.trim()}”.',
        onRetry: _load,
        secondary: TextButton(
          onPressed: _openTypedId,
          child: const Text('Ask about that ID anyway'),
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _load,
      child: ListView.separated(
        padding: const EdgeInsets.only(bottom: 24),
        itemCount: rows.length,
        separatorBuilder: (_, _) => const Divider(height: 1),
        itemBuilder: (context, i) {
          final a = rows[i];
          final species = a['species']?.toString() ?? '';
          return ListTile(
            leading: CircleAvatar(
              backgroundColor: Colors.grey.shade200,
              child: Icon(
                species == 'buffalo' ? Icons.water_drop_outlined : Icons.pets,
                size: 20,
                color: Colors.grey.shade700,
              ),
            ),
            title: Text(a['breed']?.toString() ?? 'Unknown breed'),
            subtitle: Text(
              '${a['animal_id']} · ${a['village'] ?? 'unknown village'}',
            ),
            trailing: const Icon(Icons.chat_bubble_outline, size: 20),
            onTap: () => _openChat(a),
          );
        },
      ),
    );
  }
}

class _Retry extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;
  final Widget? secondary;

  const _Retry({required this.message, required this.onRetry, this.secondary});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.cloud_off, size: 40, color: Colors.grey.shade500),
            const SizedBox(height: 12),
            Text(
              message,
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey.shade700),
            ),
            const SizedBox(height: 16),
            OutlinedButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh),
              label: const Text('Try again'),
            ),
            if (secondary != null) secondary!,
          ],
        ),
      ),
    );
  }
}
