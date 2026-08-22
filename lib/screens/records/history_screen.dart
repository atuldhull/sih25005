import 'dart:convert';

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

import '../../models/scoring_result.dart';
import '../../services/api_service.dart';
import '../../services/db_service.dart';
import '../../services/history_cache_service.dart';
import '../explain/scorecard_screen.dart';

/// Displays an animal's past scoring sessions with a weight trend chart.
///
/// Data loading order:
/// 1. Try [ApiService.getHistory] — if successful, display live data and cache it.
/// 2. If the API fails, read from [HistoryCacheService] — display cached
///    history with an offline indicator.
/// 3. If neither is available, show "No history available offline yet."
///
/// Tapping a past session opens the real ScorecardScreen (Step 4.3) from the
/// locally stored SQLite `result_json` when a valid local result exists.
class HistoryScreen extends StatefulWidget {
  /// The animal/tag id whose history is displayed.
  final String animalId;

  const HistoryScreen({super.key, required this.animalId});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

enum _LoadState { loading, live, cached, noData, error }

class _HistoryScreenState extends State<HistoryScreen> {
  _LoadState _state = _LoadState.loading;
  List<_HistoryEntry> _entries = const [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _state = _LoadState.loading);

    try {
      final result = await ApiService().getHistory(widget.animalId);

      if (_isErrorMap(result)) {
        await _loadFromCache();
        return;
      }

      final parsed = _parseHistory(result);
      if (!mounted) return;

      if (parsed.isEmpty) {
        setState(() {
          _entries = const [];
          _state = _LoadState.noData;
        });
        return;
      }

      // Live data succeeded — cache it for offline fallback.
      await HistoryCacheService.saveHistory(
        widget.animalId,
        parsed.map((e) => e.raw).toList(),
      );
      if (!mounted) return;

      setState(() {
        _entries = parsed;
        _state = _LoadState.live;
      });
    } catch (_) {
      await _loadFromCache();
    }
  }

  /// Attempts to display cached history after a live failure.
  Future<void> _loadFromCache() async {
    try {
      final cached = await HistoryCacheService.getHistory(widget.animalId);
      if (!mounted) return;

      final parsed = _parseHistory({'cached_sessions': cached});
      if (parsed.isEmpty) {
        setState(() {
          _entries = const [];
          _state = _LoadState.noData;
        });
      } else {
        setState(() {
          _entries = parsed;
          _state = _LoadState.cached;
        });
      }
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _entries = const [];
        _state = _LoadState.error;
      });
    }
  }

  /// Detects whether an ApiService result map represents a failure.
  ///
  /// ApiService returns `{'error': ...}` on any failure (network, timeout,
  /// non-2xx, malformed JSON). A real history response must NOT contain
  /// an `error` key.
  bool _isErrorMap(Map<String, dynamic> map) {
    final error = map['error'];
    return error is String && error.isNotEmpty;
  }

  /// Safely extracts a list of history entries from an ApiService response.
  List<_HistoryEntry> _parseHistory(Map<String, dynamic> map) {
    final records = _extractSessionList(map);

    final entries = <_HistoryEntry>[];
    for (final record in records) {
      final date = _extractDate(record);
      final weight = _extractWeight(record);
      entries.add(_HistoryEntry(date: date, weightKg: weight, raw: record));
    }

    // Sort chronologically (oldest first) so the trend chart reads
    // left-to-right over time. Records without a parseable date keep
    // their server order.
    entries.sort((a, b) {
      final d1 = a.dateTime;
      final d2 = b.dateTime;
      if (d1 != null && d2 != null) return d1.compareTo(d2);
      if (d1 != null) return -1;
      if (d2 != null) return 1;
      return 0;
    });

    return entries;
  }

  /// Extracts a list of session records from the response map.
  ///
  /// Supports common wrapper keys (`sessions`, `history`, `data`, `results`,
  /// `cached_sessions`) as well as the case where the response itself is a
  /// single session object. Never throws.
  List<Map<String, dynamic>> _extractSessionList(Map<String, dynamic> map) {
    const listKeys = [
      'sessions',
      'history',
      'data',
      'results',
      'items',
      'cached_sessions',
    ];

    for (final key in listKeys) {
      final value = map[key];
      if (value is List) {
        return value.whereType<Map<String, dynamic>>().toList();
      }
    }

    // Unknown shape — check if the map itself looks like a single session.
    final hasDate = map.keys.any(
      (k) => _dateKeys.contains(k) && map[k] != null,
    );
    final hasWeight = map.keys.any(
      (k) => _weightKeys.contains(k) && map[k] != null,
    );
    if (hasDate || hasWeight) {
      return [map];
    }

    return const [];
  }

  static const _dateKeys = [
    'captured_at',
    'date',
    'created_at',
    'session_date',
    'timestamp',
  ];

  static const _weightKeys = [
    // The one the server actually sends. GET /animal/{id}/history returns
    // sessions shaped {session_id, date, weight_kg_mid, health_flags}, and
    // weight_kg_mid was missing from this list - so _extractWeight returned
    // null for every session, the trend read "No weight data available", and
    // every row showed a dash. Verified on the emulator against live data
    // where the server was returning weight_kg_mid: 426.
    'weight_kg_mid',
    'weight_kg',
    'weight',
    'weight_low',
    'weight_high',
    'low',
    'high',
  ];

  /// Extracts a displayable date string. Returns the raw server string if
  /// it cannot be parsed, or `null` if no date field exists.
  String? _extractDate(Map<String, dynamic> record) {
    for (final key in _dateKeys) {
      final value = record[key];
      if (value is String && value.trim().isNotEmpty) {
        return value.trim();
      }
    }
    return null;
  }

  /// Extracts a weight in kg, preferring `{low, high}` midpoint then a
  /// direct numeric value. Returns `null` when no weight is present.
  double? _extractWeight(Map<String, dynamic> record) {
    // weight_kg_mid first: it is what the history endpoint sends, and it is
    // already a midpoint scalar rather than a {low, high} pair.
    // Nested weight_kg / weight maps: {low, high} or direct number.
    for (final key in ['weight_kg_mid', 'weight_kg', 'weight']) {
      final value = record[key];
      final direct = _asDouble(value);
      if (direct != null) return direct;
      if (value is Map) {
        final low = _asDouble(value['low']);
        final high = _asDouble(value['high']);
        if (low != null && high != null) {
          return (low + high) / 2;
        }
        if (low != null) return low;
        if (high != null) return high;
      }
    }

    // Flat low/high fields on the record itself.
    final low = _asDouble(record['weight_low']) ?? _asDouble(record['low']);
    final high = _asDouble(record['weight_high']) ?? _asDouble(record['high']);
    if (low != null && high != null) {
      return (low + high) / 2;
    }
    if (low != null) return low;
    if (high != null) return high;

    return null;
  }

  /// Converts a dynamic value to a double when it is numeric or a
  /// numeric string. Returns `null` otherwise.
  double? _asDouble(dynamic value) {
    if (value is num) return value.toDouble();
    if (value is String) return double.tryParse(value.trim());
    return null;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Scoring History')),
      body: switch (_state) {
        _LoadState.loading => const Center(child: CircularProgressIndicator()),
        _LoadState.live => _buildContent(isLive: true),
        _LoadState.cached => _buildContent(isLive: false),
        _LoadState.noData => const _NoHistoryView(),
        _LoadState.error => _ErrorView(onRetry: _load),
      },
    );
  }

  Widget _buildContent({required bool isLive}) {
    final chartEntries = _entries
        .where((e) => e.weightKg != null)
        .toList(growable: false);

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _HistoryStatusBanner(isLive: isLive),
        const SizedBox(height: 16),
        _WeightChartCard(entries: chartEntries),
        const SizedBox(height: 16),
        Text(
          'Past Sessions',
          style: Theme.of(
            context,
          ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 8),
        ..._entries.map(
          (e) => _HistoryRow(entry: e, animalId: widget.animalId),
        ),
      ],
    );
  }
}

/// Banner indicating whether the displayed history is live or cached.
class _HistoryStatusBanner extends StatelessWidget {
  final bool isLive;

  const _HistoryStatusBanner({required this.isLive});

  @override
  Widget build(BuildContext context) {
    final color = isLive ? Colors.green.shade700 : Colors.orange.shade800;
    final icon = isLive ? Icons.cloud_done_outlined : Icons.cloud_off_outlined;
    final label = isLive ? 'LIVE HISTORY' : 'OFFLINE — LAST CACHED HISTORY';

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.10),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.4)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: color),
          const SizedBox(width: 8),
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: color,
              letterSpacing: 0.5,
            ),
          ),
        ],
      ),
    );
  }
}

/// Line chart of weight over sessions.
class _WeightChartCard extends StatelessWidget {
  /// Entries that have a non-null weight, already in chronological order.
  final List<_HistoryEntry> entries;

  const _WeightChartCard({required this.entries});

  /// The subset that was actually measured, in the order given.
  List<_HistoryEntry> get measured =>
      entries.where((e) => e.plottableWeight != null).toList();

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: Colors.grey.shade300),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Weight Trend',
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 4),
            Text(
              'Weight in kg',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
            const SizedBox(height: 16),

            // ONLY MEASURED SESSIONS ARE PLOTTED.
            //
            // The baseline engine invents a weight per animal from a fixed
            // seed, so those figures are stable and rise and fall like real
            // ones. Plotted, three of them became a confident upward trend
            // for an animal nothing had ever weighed - while the assistant,
            // reading the same sessions, correctly refused to give a weight
            // at all. A line going up is more persuasive than a paragraph
            // saying there is no measurement, so the line has to go.
            if (measured.isEmpty)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 24),
                child: Center(
                  child: Text(
                    entries.isEmpty
                        ? 'No weight data available.'
                        : 'No measured weights yet. The sessions on record '
                              'ran on the demonstration engine, so there is '
                              'nothing real to plot.',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 14, color: Colors.grey.shade600),
                  ),
                ),
              )
            else ...[
              SizedBox(height: 220, child: _buildChart(measured)),
              if (measured.length < entries.length) ...[
                const SizedBox(height: 10),
                Text(
                  '${entries.length - measured.length} of ${entries.length} '
                  'sessions are not plotted: they ran on the demonstration '
                  'engine and produced no real measurement.',
                  style: TextStyle(
                    fontSize: 12,
                    height: 1.35,
                    color: Colors.grey.shade600,
                  ),
                ),
              ],
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildChart(List<_HistoryEntry> entries) {
    final spots = <FlSpot>[
      for (var i = 0; i < entries.length; i++)
        FlSpot(i.toDouble(), entries[i].weightKg!),
    ];

    double minX;
    double maxX;
    if (entries.length == 1) {
      // Single point — center the dot and prevent axis crashes.
      minX = -0.5;
      maxX = 0.5;
    } else {
      minX = 0;
      maxX = (entries.length - 1).toDouble();
    }

    // Keep the Y axis slightly padded around the data range.
    final weights = entries.map((e) => e.weightKg!).toList();
    final minWeight = weights.reduce((a, b) => a < b ? a : b);
    final maxWeight = weights.reduce((a, b) => a > b ? a : b);
    final padding = (maxWeight - minWeight) * 0.15 + 1;
    final minY = (minWeight - padding).floorToDouble();
    final maxY = (maxWeight + padding).ceilToDouble();

    return LineChart(
      LineChartData(
        minX: minX,
        maxX: maxX,
        minY: minY,
        maxY: maxY,
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: false,
            color: Colors.green.shade700,
            barWidth: 2.5,
            dotData: const FlDotData(show: true),
            belowBarData: BarAreaData(
              show: true,
              color: Colors.green.withValues(alpha: 0.10),
            ),
          ),
        ],
        gridData: const FlGridData(show: true, drawVerticalLine: false),
        borderData: FlBorderData(
          show: true,
          border: Border.all(color: Colors.grey.shade300),
        ),
        lineTouchData: LineTouchData(
          touchTooltipData: LineTouchTooltipData(
            getTooltipItems: (touchedSpots) => [
              for (final spot in touchedSpots)
                LineTooltipItem(
                  '${spot.y.toStringAsFixed(1)} kg',
                  TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
                ),
            ],
          ),
        ),
        titlesData: FlTitlesData(
          topTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
          rightTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 44,
              getTitlesWidget: (value, meta) => Text(
                value.toStringAsFixed(0),
                style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
              ),
            ),
          ),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: entries.length > 1,
              reservedSize: 28,
              interval: 1,
              getTitlesWidget: (value, meta) {
                final index = value.toInt();
                if (index < 0 || index >= entries.length) {
                  return const SizedBox.shrink();
                }
                final date = entries[index].dateTime;
                if (date == null) return const SizedBox.shrink();
                return Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: Text(
                    '${date.day}/${date.month}',
                    style: TextStyle(fontSize: 10, color: Colors.grey.shade600),
                  ),
                );
              },
            ),
          ),
        ),
      ),
    );
  }
}

/// A single history row: date + weight.
///
/// Tapping a row attempts to open the real ScorecardScreen from the locally
/// stored SQLite `result_json` for the matching session. If no valid local
/// result exists, a friendly neutral message is shown instead.
class _HistoryRow extends StatelessWidget {
  final _HistoryEntry entry;
  final String animalId;

  const _HistoryRow({required this.entry, required this.animalId});

  @override
  Widget build(BuildContext context) {
    final measured = entry.measured;
    final weightText = entry.weightKg == null
        ? '—'
        : '${entry.weightKg!.toStringAsFixed(1)} kg';

    final dateText = _formatDate(entry.date) ?? 'Date not available';

    return Card(
      elevation: 0,
      margin: const EdgeInsets.only(bottom: 8),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: Colors.grey.shade300),
      ),
      child: ListTile(
        leading: const Icon(Icons.event_note_outlined),
        title: Text(
          dateText,
          style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w500),
        ),
        subtitle: Text(
          measured ? 'Scoring session' : 'Scoring session - not measured',
          style: measured
              ? null
              : const TextStyle(color: Color(0xFFE65100)),
        ),
        trailing: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              measured ? weightText : 'placeholder',
              style: TextStyle(
                fontSize: measured ? 15 : 13,
                fontWeight: FontWeight.w600,
                // Green reads as a healthy measured figure. A number that
                // was never measured must not be dressed as one.
                color: measured
                    ? Colors.green.shade700
                    : const Color(0xFFE65100),
              ),
            ),
            Text(
              measured ? 'weight' : 'not a measurement',
              style: measured
                  ? null
                  : TextStyle(fontSize: 11, color: Colors.grey.shade600),
            ),
          ],
        ),
        onTap: () => _openScorecard(context),
      ),
    );
  }

  /// Opens the ScorecardScreen for this history row's session.
  ///
  /// Correlates the history row to a local SQLite session by animal ID and
  /// captured date, reads its `result_json`, decodes it, and navigates.
  /// Never crashes — any missing/malformed data shows a friendly message.
  Future<void> _openScorecard(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    final navigator = Navigator.of(context);

    try {
      final sessions = await DbService.getSessionsByAnimal(animalId);
      final session = _findMatchingSession(sessions);

      if (session == null) {
        messenger
          ..hideCurrentSnackBar()
          ..showSnackBar(
            const SnackBar(
              content: Text('Scorecard not available offline yet.'),
              duration: Duration(seconds: 2),
            ),
          );
        return;
      }

      final resultJson = session['result_json'] as String?;
      if (resultJson == null || resultJson.trim().isEmpty) {
        messenger
          ..hideCurrentSnackBar()
          ..showSnackBar(
            const SnackBar(
              content: Text('Scorecard not available yet.'),
              duration: Duration(seconds: 2),
            ),
          );
        return;
      }

      final decoded = jsonDecode(resultJson);
      if (decoded is! Map<String, dynamic>) {
        messenger
          ..hideCurrentSnackBar()
          ..showSnackBar(
            const SnackBar(
              content: Text('Scorecard not available offline yet.'),
              duration: Duration(seconds: 2),
            ),
          );
        return;
      }

      final scoringResult = ScoringResult.fromJson(decoded);
      final photoPath = session['side_photo_path'] as String?;

      if (!context.mounted) return;
      navigator.push(
        MaterialPageRoute<void>(
          builder: (_) => ScorecardScreen(
            result: scoringResult,
            capturedPhotoPath: photoPath,
          ),
        ),
      );
    } catch (_) {
      messenger
        ..hideCurrentSnackBar()
        ..showSnackBar(
          const SnackBar(
            content: Text('Scorecard not available offline yet.'),
            duration: Duration(seconds: 2),
          ),
        );
    }
  }

  /// Finds the local SQLite session matching this history row.
  ///
  /// Matches by animal ID (already filtered) and captured date. The local
  /// session `captured_at` and the history row date are compared on the same
  /// calendar day to avoid timezone/format mismatches. Returns `null` when
  /// no safe match exists — never guesses or crosses animals.
  Map<String, dynamic>? _findMatchingSession(
    List<Map<String, dynamic>> sessions,
  ) {
    if (sessions.isEmpty) return null;

    final historyDate = entry.dateTime;

    // Prefer an exact captured_at match first.
    for (final session in sessions) {
      final capturedAt = session['captured_at'] as String?;
      if (capturedAt == null || capturedAt.isEmpty) continue;
      if (capturedAt == entry.date) return session;
    }

    // Fall back to same-day matching when formats differ.
    if (historyDate != null) {
      for (final session in sessions) {
        final capturedAt = session['captured_at'] as String?;
        final sessionDate = capturedAt == null
            ? null
            : DateTime.tryParse(capturedAt);
        if (sessionDate == null) continue;
        if (_sameDay(sessionDate, historyDate)) return session;
      }
    }

    return null;
  }

  bool _sameDay(DateTime a, DateTime b) {
    return a.year == b.year && a.month == b.month && a.day == b.day;
  }

  /// Formats a date string for display.
  String? _formatDate(String? raw) {
    if (raw == null) return null;
    final dateTime = DateTime.tryParse(raw);
    if (dateTime == null) return raw;
    final local = dateTime.toLocal();
    final day = local.day.toString().padLeft(2, '0');
    final month = local.month.toString().padLeft(2, '0');
    final year = local.year;
    return '$day/$month/$year';
  }
}

/// A parsed history entry with optional date and weight.
class _HistoryEntry {
  final String? date;
  final double? weightKg;
  final Map<String, dynamic> raw;

  /// Whether this session's figures were measured from photographs.
  ///
  /// Absent means NOT measured. An older server build does not send the
  /// field at all, and defaulting an unknown to "measured" is exactly the
  /// mistake this exists to prevent.
  bool get measured => raw['measured'] == true;

  /// A weight worth plotting: present, and actually measured.
  double? get plottableWeight => measured ? weightKg : null;

  _HistoryEntry({this.date, this.weightKg, required this.raw});

  DateTime? get dateTime => date == null ? null : DateTime.tryParse(date!);
}

/// Shown when neither live nor cached history is available.
class _NoHistoryView extends StatelessWidget {
  const _NoHistoryView();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.history_toggle_off,
              size: 48,
              color: Colors.grey.shade400,
            ),
            const SizedBox(height: 12),
            const Text(
              'No history available offline yet.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }
}

/// Shown when an unexpected error occurs and no cache fallback exists.
class _ErrorView extends StatelessWidget {
  final VoidCallback onRetry;

  const _ErrorView({required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.error_outline, size: 48, color: Colors.grey.shade400),
            const SizedBox(height: 12),
            const Text(
              'Unable to load history.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),
            OutlinedButton(onPressed: onRetry, child: const Text('Retry')),
          ],
        ),
      ),
    );
  }
}
