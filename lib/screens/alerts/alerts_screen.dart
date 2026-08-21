import 'package:flutter/material.dart';

import '../../services/api_service.dart';
import '../records/animal_profile_screen.dart';

/// The veterinary officer's escalation feed.
///
/// Every entry here is a request for a person to look at an animal, so the
/// screen's job is to make two things unmissable: which animals, and whether
/// the finding behind them was measured at all.
class AlertsScreen extends StatefulWidget {
  const AlertsScreen({super.key});

  @override
  State<AlertsScreen> createState() => _AlertsScreenState();
}

class _AlertsScreenState extends State<AlertsScreen> {
  final ApiService _api = ApiService();

  List<Map<String, dynamic>> _alerts = const [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _api.close();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    final reply = await _api.getAlerts();
    if (!mounted) return;

    final error = reply['error'];
    setState(() {
      _loading = false;
      if (error != null) {
        _error = error.toString();
        return;
      }
      _alerts = (reply['alerts'] as List? ?? [])
          .whereType<Map>()
          .map((a) => Map<String, dynamic>.from(a))
          .toList();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Vet alerts'),
        actions: [
          IconButton(
            tooltip: 'Reload',
            onPressed: _loading ? null : _load,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: _body(),
    );
  }

  Widget _body() {
    if (_loading) return const Center(child: CircularProgressIndicator());

    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.cloud_off, size: 40, color: Colors.grey.shade500),
              const SizedBox(height: 12),
              Text(
                _error!,
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.grey.shade700),
              ),
              const SizedBox(height: 8),
              Text(
                'Alerts live on the server — there is no offline copy.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 12.5, color: Colors.grey.shade600),
              ),
              const SizedBox(height: 16),
              OutlinedButton.icon(
                onPressed: _load,
                icon: const Icon(Icons.refresh),
                label: const Text('Try again'),
              ),
            ],
          ),
        ),
      );
    }

    if (_alerts.isEmpty) {
      return RefreshIndicator(
        onRefresh: _load,
        child: ListView(
          children: [
            SizedBox(height: MediaQuery.of(context).size.height * 0.25),
            Icon(
              Icons.notifications_none,
              size: 44,
              color: Colors.grey.shade400,
            ),
            const SizedBox(height: 12),
            Center(
              child: Text(
                'No animals escalated.',
                style: TextStyle(color: Colors.grey.shade700),
              ),
            ),
            const SizedBox(height: 6),
            Center(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 40),
                child: Text(
                  'An empty feed means nothing was escalated — not that every '
                  'animal is well.',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 12.5,
                    color: Colors.grey.shade600,
                    height: 1.3,
                  ),
                ),
              ),
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _load,
      child: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: _alerts.length,
        itemBuilder: (context, i) => _AlertCard(alert: _alerts[i]),
      ),
    );
  }
}

class _AlertCard extends StatefulWidget {
  final Map<String, dynamic> alert;

  const _AlertCard({required this.alert});

  @override
  State<_AlertCard> createState() => _AlertCardState();
}

class _AlertCardState extends State<_AlertCard> {
  bool _expanded = false;

  /// Absent means demonstration.
  ///
  /// An invented `skin_nodules` finding at confidence 0.82 once flowed through
  /// the knowledge graph into a real entry in a veterinary officer's feed,
  /// about an animal nothing had examined. Somebody could drive to a farm over
  /// that, so anything not explicitly marked as measured is marked as not.
  bool get _isDemonstration => widget.alert['demonstration'] != false;

  @override
  Widget build(BuildContext context) {
    final a = widget.alert;
    final animalId = a['animal_id']?.toString() ?? 'unknown';
    final village = a['village']?.toString() ?? 'unknown village';
    final date = a['date']?.toString() ?? '';
    final risks = (a['top_risks'] as List? ?? [])
        .map((e) => e.toString())
        .toList();
    final herd = (a['herd_alerts'] as List? ?? []).whereType<Map>().toList();
    final report = a['report_vet']?.toString();

    return Card(
      elevation: 0,
      margin: const EdgeInsets.only(bottom: 12),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(
          color: _isDemonstration
              ? const Color(0xFFD32F2F)
              : Colors.grey.shade300,
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (_isDemonstration) ...[
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: const Color(0xFFFDECEA),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    const Icon(
                      Icons.warning_amber_rounded,
                      size: 16,
                      color: Color(0xFFB71C1C),
                    ),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        'DEMONSTRATION ALERT — do not act on it. No trained '
                        'symptom detector examined this animal.',
                        style: TextStyle(
                          fontSize: 12,
                          height: 1.3,
                          fontWeight: FontWeight.w600,
                          color: Colors.red.shade900,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 10),
            ],

            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        animalId,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        village,
                        style: TextStyle(
                          fontSize: 13,
                          color: Colors.grey.shade700,
                        ),
                      ),
                    ],
                  ),
                ),
                if (date.isNotEmpty)
                  Text(
                    date,
                    style: TextStyle(
                      fontSize: 12.5,
                      color: Colors.grey.shade600,
                    ),
                  ),
              ],
            ),

            if (risks.isNotEmpty) ...[
              const SizedBox(height: 10),
              Wrap(
                spacing: 6,
                runSpacing: 4,
                children: risks
                    .map(
                      (r) => Chip(
                        label: Text(r, style: const TextStyle(fontSize: 12)),
                        visualDensity: VisualDensity.compact,
                        materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                        backgroundColor: Colors.grey.shade100,
                        side: BorderSide(color: Colors.grey.shade300),
                      ),
                    )
                    .toList(),
              ),
            ],

            // A herd signal is a different kind of claim from a single sick
            // animal, and is the one an officer acts on fastest.
            for (final h in herd) ...[
              const SizedBox(height: 10),
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: const Color(0xFFFFF3E0),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: const Color(0xFFEF6C00)),
                ),
                child: Row(
                  children: [
                    const Icon(
                      Icons.groups_outlined,
                      size: 18,
                      color: Color(0xFFE65100),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Possible outbreak: ${h['symptom']} in '
                        '${h['village']} — ${h['animals_affected_14d']} '
                        'animals in 14 days',
                        style: const TextStyle(
                          fontSize: 12.5,
                          height: 1.3,
                          color: Color(0xFFE65100),
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],

            if (report != null && report.trim().isNotEmpty) ...[
              const SizedBox(height: 6),
              InkWell(
                onTap: () => setState(() => _expanded = !_expanded),
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  child: Row(
                    children: [
                      Icon(
                        _expanded ? Icons.expand_less : Icons.expand_more,
                        size: 20,
                        color: Colors.grey.shade700,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        _expanded
                            ? 'Hide the screening summary'
                            : 'Read the screening summary',
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w500,
                          color: Colors.grey.shade800,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              if (_expanded)
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade50,
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: Colors.grey.shade300),
                  ),
                  child: Text(
                    report,
                    style: const TextStyle(fontSize: 13, height: 1.45),
                  ),
                ),
            ],

            const SizedBox(height: 4),
            Align(
              alignment: Alignment.centerRight,
              child: TextButton.icon(
                onPressed: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (_) => AnimalProfileScreen(animalId: animalId),
                  ),
                ),
                icon: const Icon(Icons.folder_outlined, size: 18),
                label: const Text('Open record'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
